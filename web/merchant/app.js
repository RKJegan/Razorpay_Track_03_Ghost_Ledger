/*
 * Ghost Ledger merchant dashboard.
 *
 * Talks only to this origin (relative URLs). The API key lives in sessionStorage
 * for this tab and is sent as a bearer token. All data is written with
 * textContent, never innerHTML, so API content cannot inject markup.
 */
(function () {
  "use strict";

  var KEY_NAME = "gl_merchant_key";
  var REFRESH_MS = 2000;
  var inFlight = false;
  var timer = null;
  var selectedId = null;

  var STAGE_LABELS = {
    diagnosis_created: "Diagnosed the failure",
    policy_check_passed: "Policy check passed",
    policy_denied: "Policy check declined",
    approval_requested: "Waiting for approval",
    approval_granted: "Approved",
    approval_rejected: "Approval declined",
    payment_link_created: "Payment link sent",
    payment_link_expired: "Payment link expired",
    payment_captured: "Payment received",
    payment_failed: "Customer payment failed",
    settlement_confirmed: "Settlement confirmed",
    link_call_failed: "Retrying link creation",
    reminder_queued: "Reminder queued",
    stopping_rule_triggered: "Stopped: escalated for review",
    recovery_failed: "Recovery failed"
  };

  var money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

  function $(id) { return document.getElementById(id); }

  function label(stage) { return STAGE_LABELS[stage] || stage || "—"; }

  function when(ts) {
    if (!ts) return "—";
    return String(ts).replace("T", " ").slice(0, 16);
  }

  function setText(id, text) { $(id).textContent = text; }

  function cell(text, className) {
    var td = document.createElement("td");
    td.textContent = text == null || text === "" ? "—" : String(text);
    if (className) td.className = className;
    return td;
  }

  function emptyRow(tbody, columns, text) {
    var tr = document.createElement("tr");
    var td = document.createElement("td");
    td.colSpan = columns;
    td.className = "empty";
    td.textContent = text;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  function fill(tbodyId, columns, rows, emptyText, makeRow) {
    var tbody = $(tbodyId);
    tbody.textContent = "";
    if (!rows || rows.length === 0) {
      emptyRow(tbody, columns, emptyText);
      return;
    }
    rows.forEach(function (row) { tbody.appendChild(makeRow(row)); });
  }

  function clickableRow(recoveryId, cells) {
    var tr = document.createElement("tr");
    tr.className = "clickable";
    tr.tabIndex = 0;
    tr.setAttribute("role", "button");
    tr.setAttribute("aria-label", "Show timeline for " + recoveryId);
    cells.forEach(function (c) { tr.appendChild(c); });
    var open = function () { selectRecovery(recoveryId); };
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
    });
    return tr;
  }

  function idCell(id) { return cell(id); }

  // --- network -----------------------------------------------------------

  function key() { return sessionStorage.getItem(KEY_NAME); }

  function api(path) {
    return fetch(path, {
      cache: "no-store",
      headers: { "Authorization": "Bearer " + key(), "Accept": "application/json" }
    }).then(function (resp) {
      if (resp.status === 401) {
        var err = new Error("unauthorized");
        err.code = 401;
        throw err;
      }
      if (!resp.ok) {
        throw new Error("request failed (" + resp.status + ")");
      }
      return resp.json();
    });
  }

  // --- rendering ---------------------------------------------------------

  function renderSummary(s) {
    var c = s.counts || {};
    setText("m-active", c.active);
    setText("m-failures", c.failures_in_progress);
    setText("m-approvals", c.pending_approvals);
    setText("m-settled", c.settled_today);
    setText("m-recovered", money.format(s.settled_today_inr || 0));

    fill("t-active", 5, s.active, "No recoveries in progress.", function (r) {
      return clickableRow(r.recovery_id, [
        idCell(r.recovery_id), cell(money.format(r.amount_inr)), cell(r.cause),
        cell(label(r.latest_stage)), cell(when(r.latest_at || r.updated_at))
      ]);
    });

    fill("t-failures", 4, s.failures, "No failures in progress.", function (r) {
      return clickableRow(r.recovery_id, [
        idCell(r.recovery_id), cell(money.format(r.amount_inr)),
        cell(label(r.failure_stage), "failed-text"), cell(when(r.failure_at))
      ]);
    });

    fill("t-approvals", 4, s.approvals, "Nothing waiting for approval.", function (r) {
      return clickableRow(r.recovery_id, [
        idCell(r.recovery_id), cell(money.format(r.amount_inr)), cell(r.cause), cell(when(r.created_at))
      ]);
    });
  }

  function renderAll(rows) {
    fill("t-all", 5, rows, "No recoveries match this filter.", function (r) {
      return clickableRow(r.recovery_id, [
        idCell(r.recovery_id), cell(money.format(r.amount_inr)), cell(r.cause),
        cell(r.status, r.status === "settled" ? "ok-text" : (r.status === "escalated" ? "failed-text" : "")),
        cell(when(r.updated_at))
      ]);
    });
  }

  function renderTimeline(data) {
    var list = $("timeline");
    list.textContent = "";
    data.timeline.forEach(function (ev) {
      var li = document.createElement("li");
      var stage = document.createElement("span");
      stage.className = "stage";
      stage.textContent = label(ev.stage);
      var time = document.createElement("span");
      time.className = "when";
      time.textContent = when(ev.timestamp);
      li.appendChild(stage);
      li.appendChild(time);
      var keys = Object.keys(ev.detail || {});
      if (keys.length) {
        var d = document.createElement("div");
        d.className = "detail";
        d.textContent = keys.map(function (k) { return k + ": " + ev.detail[k]; }).join(" · ");
        li.appendChild(d);
      }
      list.appendChild(li);
    });
    if (data.timeline.length === 0) {
      var none = document.createElement("li");
      none.textContent = "No events yet.";
      list.appendChild(none);
    }
    setText("detail-title", "Recovery " + data.recovery.recovery_id + " · " + data.recovery.status);
    $("detail").hidden = false;
  }

  function selectRecovery(id) {
    selectedId = id;
    api("/api/merchant/recoveries/" + encodeURIComponent(id)).then(renderTimeline).catch(function (err) {
      if (err.code === 401) return signedOut("Your key was rejected. Please connect again.");
      setText("status", "Could not load that recovery: " + err.message);
    });
  }

  // --- polling -----------------------------------------------------------

  function tick() {
    if (inFlight || !key()) return;
    inFlight = true;
    var filter = $("filter").value;
    var qs = "?limit=50" + (filter ? "&status=" + encodeURIComponent(filter) : "");
    Promise.all([
      api("/api/merchant/summary?limit=50"),
      api("/api/merchant/recoveries" + qs)
    ]).then(function (out) {
      renderSummary(out[0]);
      renderAll(out[1].items);
      setText("all-total", out[1].total + " matching recoveries");
      setText("status", "Live · updated " + when(out[0].generated_at) + " · every 2s");
      if (selectedId) { selectRecovery(selectedId); }
    }).catch(function (err) {
      if (err.code === 401) {
        signedOut("Your key was rejected. Please connect again.");
      } else {
        setText("status", "Connection problem, retrying: " + err.message);
      }
    }).then(function () { inFlight = false; });
  }

  function showDashboard(merchantName) {
    $("login").hidden = true;
    $("dashboard").hidden = false;
    $("signout").hidden = false;
    setText("who", merchantName || "");
    if (!timer) {
      tick();
      timer = setInterval(tick, REFRESH_MS);
    }
  }

  function signedOut(message) {
    sessionStorage.removeItem(KEY_NAME);
    if (timer) { clearInterval(timer); timer = null; }
    selectedId = null;
    $("dashboard").hidden = true;
    $("signout").hidden = true;
    $("detail").hidden = true;
    $("login").hidden = false;
    setText("who", "");
    setText("login-error", message || "");
    $("key").value = "";
    $("key").focus();
  }

  // --- wiring ------------------------------------------------------------

  $("login-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var typed = $("key").value.trim();
    if (!typed) return;
    sessionStorage.setItem(KEY_NAME, typed);
    setText("login-error", "");
    api("/api/merchant/me").then(function (me) {
      showDashboard(me.name || me.merchant_id);
    }).catch(function (err) {
      signedOut(err.code === 401 ? "That key is not recognised." : "Could not reach the server: " + err.message);
    });
  });

  $("signout").addEventListener("click", function () { signedOut(""); });

  $("filter").addEventListener("change", function () { tick(); });

  if (key()) {
    api("/api/merchant/me").then(function (me) {
      showDashboard(me.name || me.merchant_id);
    }).catch(function () { signedOut(""); });
  } else {
    $("login").hidden = false;
    $("key").focus();
  }
})();
