#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <rtc/rtc.hpp>
#include <rtc/websocketserver.hpp>
#include <rtc/h265rtppacketizer.hpp>
#include <rtc/rtcpsrreporter.hpp>
#include <rtc/rtcpnackresponder.hpp>
#include <rtc/plihandler.hpp>

class WebRtcStreamer {
public:
    WebRtcStreamer() : m_port(8080) {
        rtc::InitLogger(rtc::LogLevel::Warning);
    }

    ~WebRtcStreamer() {
        if (m_ws_server) m_ws_server->stop();
    }

    bool start(int port = 8080) {
        try {
            m_port = port;
            rtc::WebSocketServer::Configuration ws_config;
            ws_config.port = m_port;
            ws_config.enableTls = false; // 局域网内直接用 ws:// 即可

            m_ws_server = std::make_shared<rtc::WebSocketServer>(ws_config);

            // 当有浏览器通过 WebSocket 连接过来时
            m_ws_server->onClient([this](std::shared_ptr<rtc::WebSocket> ws) {
                std::cout << "[INFO] EagleEye: Browser connected! Initiating H.265 Offer..." << std::endl;

                rtc::Configuration rtc_config;
                // rtc_config.iceServers.emplace_back("stun:stun.l.google.com:19302");
                auto pc = std::make_shared<rtc::PeerConnection>(rtc_config);

                // 和输入码流保持一致：只协商 H.265
                rtc::Description::Video video_desc("EagleEye", rtc::Description::Direction::SendOnly);
                video_desc.addH265Codec(96);
                auto video_track = pc->addTrack(video_desc);
                auto rtp_config = std::make_shared<rtc::RtpPacketizationConfig>(1024, "EagleEye", 96, 90000, 0);
                auto packetizer = std::make_shared<rtc::H265RtpPacketizer>(
                    rtc::NalUnit::Separator::StartSequence,
                    rtp_config,
                    1200
                );
                auto sr_reporter = std::make_shared<rtc::RtcpSrReporter>(rtp_config);
                auto nack_responder = std::make_shared<rtc::RtcpNackResponder>();
                auto pli_handler = std::make_shared<rtc::PliHandler>([this]() {
                    m_need_keyframe.store(true, std::memory_order_relaxed);
                });
                packetizer->addToChain(sr_reporter);
                sr_reporter->addToChain(nack_responder);
                nack_responder->addToChain(pli_handler);
                video_track->setMediaHandler(packetizer);

                pc->onLocalCandidate([ws](rtc::Candidate candidate) {
                    std::string msg = "{\"type\":\"candidate\",\"candidate\":\"" + candidate.candidate() + "\",\"mid\":\"" + candidate.mid() + "\"}";
                    ws->send(msg);
                });

                pc->onStateChange([](rtc::PeerConnection::State state) {
                    std::cout << "[INFO] EagleEye: WebRTC state -> " << state << std::endl;
                });

                // 只要本地 SDP 生成好，立刻作为 Offer 发给浏览器
                pc->onLocalDescription([ws](rtc::Description description) {
                    std::string sdp = description.generateSdp();
                    replaceString(sdp, "\r\n", "\\r\\n");
                    std::string reply = "{\"type\":\"offer\",\"sdp\":\"" + sdp + "\"}";
                    ws->send(reply);
                });

                // 收到浏览器的 Answer / Candidate
                ws->onMessage([pc](std::variant<rtc::binary, rtc::string> data) {
                    if (std::holds_alternative<rtc::string>(data)) {
                        std::string msg = std::get<rtc::string>(data);
                        std::string type = extractValue(msg, "type");

                        if (type == "answer") {
                            std::string sdp = extractValue(msg, "sdp");
                            replaceString(sdp, "\\r\\n", "\r\n");
                            pc->setRemoteDescription(rtc::Description(sdp, type));
                        } else if (type == "candidate") {
                            std::string candidate_str = extractValue(msg, "candidate");
                            std::string mid = extractValue(msg, "mid");
                            if (!candidate_str.empty() && !mid.empty()) {
                                pc->addRemoteCandidate(rtc::Candidate(candidate_str, mid));
                            }
                        }
                    }
                });

                // 触发主动生成 Offer
                pc->setLocalDescription();

                std::lock_guard<std::mutex> lock(m_track_mutex);
                m_pc = pc;
                m_video_track = video_track;
            });

            std::cout << "[INFO] EagleEye: WebRTC Signaling Server running on ws://0.0.0.0:" << m_port << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] EagleEye: Failed to start WebRTC server on port " << port
                      << ", reason: " << e.what() << std::endl;
            m_ws_server.reset();
            return false;
        } catch (...) {
            std::cerr << "[ERROR] EagleEye: Failed to start WebRTC server on port " << port
                      << ", unknown exception" << std::endl;
            m_ws_server.reset();
            return false;
        }
    }

    // 将 VPU 编码出的 H.265 NALU 发送给浏览器
    // 关键：为每帧设置正确的 RTP timestamp，避免浏览器抖动缓冲/播放时钟异常
    bool pushFrame(const std::vector<uint8_t>& nalu, uint64_t pts_us) {
        // static FILE* fp_debug = fopen("/tmp/webrtc_debug.h265", "wb");
        // if (fp_debug && !nalu.empty()) fwrite(nalu.data(), 1, nalu.size(), fp_debug);
        
        std::shared_ptr<rtc::Track> track;
        {
            std::lock_guard<std::mutex> lock(m_track_mutex);
            track = m_video_track;
        }

        if (!track || !track->isOpen() || nalu.empty()) {
            return false;
        }

        // H.265 视频 RTP 时钟频率固定 90kHz
        static constexpr uint64_t kVideoClock = 90000ULL;
        static constexpr uint64_t kUsPerSecond = 1000000ULL;
        const uint32_t rtp_ts = static_cast<uint32_t>((pts_us * kVideoClock) / kUsPerSecond);
        track->sendFrame(reinterpret_cast<const std::byte*>(nalu.data()),
                         nalu.size(),
                         rtc::FrameInfo(rtp_ts));

        if (m_sent_frames > 0) {
            const uint32_t rtp_step = rtp_ts - m_last_rtp_ts;
            if (m_rtp_step_ema <= 0.0) {
                m_rtp_step_ema = static_cast<double>(rtp_step);
            } else {
                m_rtp_step_ema = m_rtp_step_ema * 0.9 + static_cast<double>(rtp_step) * 0.1;
            }
            m_rtp_step_samples++;

            if (m_rtp_step_samples > 30) {
                const double low = m_rtp_step_ema * 0.5;
                const double high = m_rtp_step_ema * 1.5;
                if (static_cast<double>(rtp_step) < low || static_cast<double>(rtp_step) > high) {
                    m_rtp_anomaly_count++;
                    if ((m_rtp_anomaly_count % 20) == 1) {
                        std::cout << "[WARN] EagleEye: RTP timestamp step abnormal: "
                                  << rtp_step << " ticks (ema="
                                  << static_cast<uint32_t>(m_rtp_step_ema)
                                  << ", count=" << m_rtp_anomaly_count
                                  << ")" << std::endl;
                    }
                }
            }
        }
        m_last_rtp_ts = rtp_ts;
        m_sent_frames++;

        return true;
    }

    bool consumeKeyframeRequest() {
        return m_need_keyframe.exchange(false, std::memory_order_relaxed);
    }

private:
    std::shared_ptr<rtc::WebSocketServer> m_ws_server;
    std::shared_ptr<rtc::PeerConnection> m_pc;
    std::shared_ptr<rtc::Track> m_video_track;
    std::mutex m_track_mutex;
    std::atomic<bool> m_need_keyframe{false};
    uint64_t m_sent_frames{0};
    uint32_t m_last_rtp_ts{0};
    uint64_t m_rtp_anomaly_count{0};
    double m_rtp_step_ema{0.0};
    uint64_t m_rtp_step_samples{0};
    int m_port;

    // 极简字符串辅助函数 (代替沉重的 JSON 库)
    static std::string extractValue(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":\"";
        auto pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos += search.length();
        auto end = json.find("\"", pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }

    static void replaceString(std::string& subject, const std::string& search, const std::string& replace) {
        size_t pos = 0;
        while ((pos = subject.find(search, pos)) != std::string::npos) {
            subject.replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }
};
