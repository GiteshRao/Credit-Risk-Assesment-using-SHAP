(() => {
  const form = document.getElementById("riskForm");
  const submitBtn = document.getElementById("submitBtn");
  const errorNote = document.getElementById("errorNote");
  const verdict = document.getElementById("verdict");

  const incomeInput = document.getElementById("person_income");
  const amountInput = document.getElementById("loan_amnt");
  const percentInput = document.getElementById("loan_percent_income");

  const gaugeFill = document.getElementById("gaugeFill");
  const gaugeThreshold = document.getElementById("gaugeThreshold");
  const probNumber = document.getElementById("probNumber");
  const stampBadge = document.getElementById("stampBadge");
  const stampText = document.getElementById("stampText");

  const factProb = document.getElementById("factProb");
  const factThreshold = document.getElementById("factThreshold");
  const factResult = document.getElementById("factResult");

  const apiDot = document.getElementById("apiDot");
  const apiStatusText = document.getElementById("apiStatusText");

  const GAUGE_CIRCUMFERENCE = 540.35; // 2 * PI * 86, matches CSS

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  /* =========================================================
     Motion layer: scroll progress, hero scrub, reveals, tilt
     ========================================================= */
  const progressBar = document.getElementById("progress");
  const nav = document.getElementById("nav");
  const hero = document.getElementById("hero");
  const cursorGlow = document.getElementById("cursorGlow");

  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const y = window.scrollY;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      progressBar.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
      nav.classList.toggle("scrolled", y > 40);

      // hero scrub: 0 at top → 1 once the sticky stage has been scrolled through
      const range = hero.offsetHeight - window.innerHeight;
      const p = Math.min(1, Math.max(0, y / range));
      hero.style.setProperty("--p", p.toFixed(4));
      ticking = false;
    });
  }
  if (!reduceMotion) {
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    onScroll();
  }

  // Scroll reveals
  const revealEls = document.querySelectorAll("[data-reveal]");
  if ("IntersectionObserver" in window && !reduceMotion) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: "0px 0px -8% 0px" }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("in"));
  }

  // 3D tilt helper (mouse-driven)
  function attachTilt(el, max, onMove, onLeave) {
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width;
      const y = (e.clientY - r.top) / r.height;
      el.style.setProperty("--ry", `${((x - 0.5) * 2 * max).toFixed(2)}deg`);
      el.style.setProperty("--rx", `${((0.5 - y) * 2 * max).toFixed(2)}deg`);
      el.style.setProperty("--mx", `${(x * 100).toFixed(1)}%`);
      el.style.setProperty("--my", `${(y * 100).toFixed(1)}%`);
      if (onMove) onMove(x, y);
    });
    el.addEventListener("pointerleave", () => {
      el.style.setProperty("--ry", "0deg");
      el.style.setProperty("--rx", "0deg");
      if (onLeave) onLeave();
    });
  }

  if (finePointer && !reduceMotion) {
    // Hero card stack follows the cursor; chips float at different depths
    const stage = document.getElementById("stage");
    const tilt = document.getElementById("stageTilt");
    const chips = stage.querySelectorAll(".chip");
    stage.addEventListener("pointermove", (e) => {
      const r = stage.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      tilt.style.setProperty("--ry", `${(x * 18).toFixed(2)}deg`);
      tilt.style.setProperty("--rx", `${(-y * 14).toFixed(2)}deg`);
      chips.forEach((c) => {
        const d = parseFloat(c.dataset.depth || "30") / 30;
        c.style.setProperty("--cx", `${(x * 26 * d).toFixed(1)}px`);
        c.style.setProperty("--cy", `${(y * 20 * d).toFixed(1)}px`);
      });
    });
    stage.addEventListener("pointerleave", () => {
      tilt.style.setProperty("--ry", "0deg");
      tilt.style.setProperty("--rx", "0deg");
      chips.forEach((c) => {
        c.style.setProperty("--cx", "0px");
        c.style.setProperty("--cy", "0px");
      });
    });

    // Layer cards
    document.querySelectorAll("[data-tilt]").forEach((el) => attachTilt(el, 9));

    // Soft cursor glow with easing
    let tx = window.innerWidth / 2, ty = window.innerHeight / 3, cx = tx, cy = ty;
    window.addEventListener("pointermove", (e) => { tx = e.clientX; ty = e.clientY; });
    (function loop() {
      cx += (tx - cx) * 0.08;
      cy += (ty - cy) * 0.08;
      cursorGlow.style.transform = `translate3d(${cx}px, ${cy}px, 0)`;
      requestAnimationFrame(loop);
    })();
  } else {
    cursorGlow.style.display = "none";
  }

  /* =========================================================
     Auto-calculate loan-to-income ratio
     ========================================================= */
  function recalcPercent() {
    const income = parseFloat(incomeInput.value);
    const amount = parseFloat(amountInput.value);
    if (income > 0 && amount >= 0) {
      percentInput.value = (amount / income).toFixed(2);
    }
  }
  incomeInput.addEventListener("input", recalcPercent);
  amountInput.addEventListener("input", recalcPercent);
  recalcPercent();

  /* =========================================================
     Service status check
     ========================================================= */
  fetch("/openapi.json", { method: "GET" })
    .then((res) => {
      if (res.ok) {
        apiDot.classList.add("ok");
        apiStatusText.textContent = "service ready";
      } else {
        throw new Error("bad status");
      }
    })
    .catch(() => {
      apiDot.classList.add("down");
      apiStatusText.textContent = "service unreachable";
    });

  /* =========================================================
     Helpers
     ========================================================= */
  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("loading", isLoading);
    submitBtn.querySelector(".btn-label").textContent = isLoading
      ? "Reviewing file…"
      : "Assess risk";
  }

  function showError(message) {
    errorNote.textContent = message;
    errorNote.hidden = false;
  }

  function clearError() {
    errorNote.hidden = true;
    errorNote.textContent = "";
  }

  function animateNumber(el, from, to, duration) {
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = from + (to - from) * eased;
      el.textContent = value.toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = to.toFixed(1);
    }
    requestAnimationFrame(tick);
  }

  function renderVerdict(data) {
    const probabilityPct = data.default_probability * 100;
    const thresholdPct = data.threshold * 100;
    const isHighRisk = data.default_prediction === 1;

    // restart the entrance animation each time
    verdict.hidden = false;
    verdict.style.animation = "none";
    void verdict.offsetWidth;
    verdict.style.animation = "";
    verdict.scrollIntoView({ behavior: "smooth", block: "center" });

    // Gauge fill: gradient for low risk, solid red when flagged
    const offset = GAUGE_CIRCUMFERENCE * (1 - probabilityPct / 100);
    gaugeFill.style.stroke = isHighRisk ? "var(--risk-red)" : "url(#gaugeGrad)";
    gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        gaugeFill.style.strokeDashoffset = offset;
      });
    });

    // Threshold tick
    gaugeThreshold.style.transform = `rotate(${thresholdPct * 3.6}deg)`;

    // Number readout
    animateNumber(probNumber, 0, probabilityPct, 1200);

    // Stamp
    stampBadge.classList.remove("stamp--in", "risk-high");
    void stampBadge.offsetWidth; // restart animation
    if (isHighRisk) {
      stampBadge.classList.add("risk-high");
      stampText.textContent = "HIGH RISK";
    } else {
      stampText.textContent = "LOW RISK";
    }
    requestAnimationFrame(() => stampBadge.classList.add("stamp--in"));

    // Ledger facts
    factProb.textContent = `${probabilityPct.toFixed(1)}%`;
    factThreshold.textContent = `${thresholdPct.toFixed(1)}%`;
    factResult.textContent = data.Result;
  }

  /* =========================================================
     Submit
     ========================================================= */
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    setLoading(true);

    const payload = {
      person_age: parseInt(document.getElementById("person_age").value, 10),
      person_income: parseFloat(incomeInput.value),
      person_home_ownership: document.getElementById("person_home_ownership").value,
      person_emp_length: parseFloat(document.getElementById("person_emp_length").value),
      loan_intent: document.getElementById("loan_intent").value,
      loan_grade: document.getElementById("loan_grade").value,
      loan_amnt: parseFloat(amountInput.value),
      loan_int_rate: parseFloat(document.getElementById("loan_int_rate").value),
      loan_percent_income: parseFloat(percentInput.value),
      cb_person_default_on_file: document.getElementById("cb_person_default_on_file").value,
      cb_person_cred_hist_length: parseInt(document.getElementById("cb_person_cred_hist_length").value, 10),
    };

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = body && body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`;
        throw new Error(detail);
      }

      const data = await res.json();
      renderVerdict(data);
    } catch (err) {
      showError(`Could not reach the ledger. ${err.message || "Check the service is running."}`);
    } finally {
      setLoading(false);
    }
  });
})();