#pragma once

#include <string_view>

// Qorvix default port allocation — the single source of truth.
//
// The 2005-2010 block is reserved for Qorvix services so a full deployment never collides with
// itself and operators can firewall one contiguous range. Defaults only: every service must still
// accept an override (--port and/or an env var), because a default that cannot be changed is a
// deployment bug waiting to happen.
//
// STATUS: only the Runtime exists today (`qorvix serve`). The rest are reserved so the services
// arrive on the numbers they were planned for instead of inventing their own — see the phase notes
// on each constant. Reserving costs nothing; renumbering a shipped service costs users.
namespace qorvix::ports {

// OpenAI-compatible inference HTTP server — `qorvix serve`. Phase 9, SHIPPED.
inline constexpr int kRuntime = 2005;

// Front door for multi-node/multi-model routing: auth, rate limiting, model-name -> backend
// dispatch. Not yet built (Phase 13 territory — API keys and rate limiting live there).
inline constexpr int kGateway = 2006;

// Web UI (React/TS/Vite dashboard: chat, models, memory, performance pages). Phase 12.
inline constexpr int kDashboard = 2007;

// Operational control plane, deliberately separate from the inference port so it can be bound to
// a private interface and firewalled independently: model load/unload, scheduler introspection,
// runtime config. Not yet built.
inline constexpr int kAdminApi = 2008;

// Prometheus scrape endpoint (/metrics). Separate from the admin port so metrics can be exposed
// to a scraper without also exposing control operations. Phase 12 (Prometheus/Grafana wiring).
inline constexpr int kMetrics = 2009;

// gRPC inference endpoint, for clients that want streaming without SSE framing overhead.
// Not yet on the roadmap; the number is reserved so it does not land on top of something else.
inline constexpr int kGrpc = 2010;

// The reserved block, inclusive. Anything Qorvix binds by default lives in here.
inline constexpr int kRangeFirst = 2005;
inline constexpr int kRangeLast = 2010;

inline constexpr bool inReservedRange(int port) noexcept {
  return port >= kRangeFirst && port <= kRangeLast;
}

// Human-readable name for a reserved port, or "" if it isn't one of ours. Useful in bind-failure
// messages ("port 2007 is the Qorvix Dashboard — is one already running?").
inline constexpr std::string_view serviceName(int port) noexcept {
  switch (port) {
    case kRuntime:   return "Qorvix Runtime";
    case kGateway:   return "Qorvix Gateway";
    case kDashboard: return "Qorvix Dashboard";
    case kAdminApi:  return "Qorvix Admin API";
    case kMetrics:   return "Qorvix Metrics";
    case kGrpc:      return "Qorvix gRPC";
    default:         return {};
  }
}

}  // namespace qorvix::ports
