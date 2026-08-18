/* ==========================================================================
   PRADEEP M - SILICON LAB PORTFOLIO JAVASCRIPT
   Interactive features: Circuit canvas, Pipeline stage selector,
   RTL DFF simulator, Project case-study modal, Terminal CLI simulator
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {

  /* ------------------------------------------------------------------------
     1. BACKGROUND CIRCUIT CANVAS ANIMATION
     ------------------------------------------------------------------------ */
  const canvas = document.getElementById('circuit-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    });

    class SignalParticle {
      constructor() {
        this.reset();
      }
      reset() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.length = Math.random() * 80 + 40;
        this.speed = Math.random() * 2 + 1;
        this.dir = Math.random() > 0.5 ? 'h' : 'v';
        this.alpha = Math.random() * 0.4 + 0.1;
      }
      update() {
        if (this.dir === 'h') {
          this.x += this.speed;
          if (this.x > width) this.reset();
        } else {
          this.y += this.speed;
          if (this.y > height) this.reset();
        }
      }
      draw() {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(0, 240, 255, ${this.alpha})`;
        ctx.lineWidth = 1;
        if (this.dir === 'h') {
          ctx.moveTo(this.x, this.y);
          ctx.lineTo(this.x + this.length, this.y);
        } else {
          ctx.moveTo(this.x, this.y);
          ctx.lineTo(this.x, this.y + this.length);
        }
        ctx.stroke();
      }
    }

    const particles = Array.from({ length: 35 }, () => new SignalParticle());

    function animate() {
      ctx.clearRect(0, 0, width, height);
      particles.forEach(p => {
        p.update();
        p.draw();
      });
      requestAnimationFrame(animate);
    }
    animate();
  }


  /* ------------------------------------------------------------------------
     2. HERO HARDWARE PIPELINE INTERACTION
     ------------------------------------------------------------------------ */
  const pipelineStages = document.querySelectorAll('.pipe-stage');
  const pipeTagName = document.getElementById('pipe-tag-name');
  const pipeTagDesc = document.getElementById('pipe-tag-desc');

  const stageData = {
    spec: {
      name: "SPECIFICATION & ARCHITECTURE",
      desc: "Analyzing protocol timing requirements, bus interfaces (AXI/APB), memory constraints, and defining block-level microarchitectural specifications."
    },
    rtl: {
      name: "RTL DESIGN (VERILOG / SYSTEMVERILOG)",
      desc: "Translating microarchitectural specs into synthesizable Verilog/SystemVerilog RTL with proper FSM state encoding, clock domain crossing (CDC), and register pipelines."
    },
    verify: {
      name: "FUNCTIONAL VERIFICATION",
      desc: "Building cycle-accurate testbenches, Python golden models, scoreboard comparison models, and ULP tolerance checkers to ensure 100% bug-free operation."
    },
    synth: {
      name: "LOGIC SYNTHESIS (VIVADO / GENUS)",
      desc: "Synthesizing RTL into gate-level netlists targeting Xilinx technology libraries, checking area LUT/FF utilization and setup/hold timing paths."
    },
    fpga: {
      name: "FPGA IMPLEMENTATION & DEBUG",
      desc: "Place and route on target FPGAs (Spartan-7, PYNQ-Z2), bitstream generation, and hardware-in-the-loop debugging using logic analyzers."
    }
  };

  pipelineStages.forEach(stage => {
    stage.addEventListener('mouseenter', () => {
      pipelineStages.forEach(s => s.classList.remove('active'));
      stage.classList.add('active');
      const key = stage.getAttribute('data-stage');
      if (stageData[key]) {
        pipeTagName.textContent = stageData[key].name;
        pipeTagDesc.textContent = stageData[key].desc;
      }
    });
  });


  /* ------------------------------------------------------------------------
     3. INTERACTIVE RTL LAB SIMULATOR (D-FLIP-FLOP)
     ------------------------------------------------------------------------ */
  let inputD = 0;
  let outputQ = 0;

  const btnToggleD = document.getElementById('btn-toggle-d');
  const btnPulseClk = document.getElementById('btn-pulse-clk');
  const btnResetRst = document.getElementById('btn-reset-rst');
  
  const valDIn = document.getElementById('val-d-in');
  const valQOut = document.getElementById('val-q-out');
  const wireD = document.getElementById('wire-d');
  const wireQ = document.getElementById('wire-q');
  const labLog = document.getElementById('lab-log');

  function appendLog(msg) {
    const time = (performance.now() / 1000).toFixed(2);
    labLog.innerHTML += `<br>[${time} ns] ${msg}`;
    labLog.scrollTop = labLog.scrollHeight;
  }

  if (btnToggleD && btnPulseClk && btnResetRst) {
    btnToggleD.addEventListener('click', () => {
      inputD = inputD === 0 ? 1 : 0;
      valDIn.textContent = inputD;
      btnToggleD.innerHTML = `<i class="fa-solid fa-toggle-${inputD ? 'on' : 'off'}"></i> Toggle D Input (Current: ${inputD})`;
      wireD.classList.remove('pulse');
      void wireD.offsetWidth;
      wireD.classList.add('pulse');
      appendLog(`Input D toggled to ${inputD}. Output Q remains ${outputQ} (waiting for CLK posedge).`);
    });

    btnPulseClk.addEventListener('click', () => {
      outputQ = inputD;
      valQOut.textContent = outputQ;
      wireQ.classList.remove('pulse');
      void wireQ.offsetWidth;
      wireQ.classList.add('pulse');
      appendLog(`⚡ POSITIVE CLK EDGE! Sampled D = ${inputD} -> Registered Q = ${outputQ}.`);
    });

    btnResetRst.addEventListener('click', () => {
      outputQ = 0;
      valQOut.textContent = 0;
      appendLog(`🔴 ACTIVE-HIGH RESET ASSERTED. Registered Q forced to 0 asynchronously.`);
    });
  }


  /* ------------------------------------------------------------------------
     4. PROJECT CASE-STUDY DASHBOARD MODALS
     ------------------------------------------------------------------------ */
  const projectModal = document.getElementById('project-modal');
  const modalContent = document.getElementById('modal-content');
  const modalCloseBtn = document.getElementById('modal-close-btn');

  const projectDetails = {
    gqa: `
      <div class="project-modal-header">
        <div class="project-badge-strip">
          <span class="hero-flag"><i class="fa-solid fa-star"></i> HERO PROJECT</span>
          <span class="project-tags">BF16 • GQA ACCELERATOR • ZYNQ-7000 • AXI4</span>
        </div>
        <h2>BF16 Grouped-Query Attention Hardware Accelerator</h2>
        <p class="modal-sub">Comprehensive Hardware Case Study Dashboard</p>
      </div>

      <div class="modal-body-content">
        <h3>1. Problem & Motivation</h3>
        <p>Transformer-based Large Language Models (e.g. Llama-3 8B) require massive memory bandwidth and execution cycles during self-attention computation. Standard full Multi-Head Attention (MHA) creates prohibitive hardware footprints on embedded edge devices like the PYNQ-Z2 (XC7Z020 FPGA).</p>

        <h3>2. Hardware Architecture</h3>
        <p>This core implements Grouped-Query Attention (GQA) with <strong>32 Q heads</strong> and <strong>8 KV heads</strong> (Group Size 4, Head Dimension 128). To fit inside the XC7Z020's limited LUT and DSP count without native BF16 hard macros, a time-multiplexed compute lane configuration (<code>GROUP_SIZE=1</code> override) was engineered.</p>
        
        <div class="arch-flow-diagram" style="margin: 1rem 0;">
          <div class="arch-node highlight">AXI4-Stream Ingress (64-bit / 256-bit DMA)</div>
          <div class="arch-arrow">↓</div>
          <div class="arch-node">RoPE Unit (Rotary Position Embedding ROM + Causal Masking)</div>
          <div class="arch-arrow">↓</div>
          <div class="arch-node highlight">BF16 MAC Datapath & Softmax Scaling Unit</div>
          <div class="arch-arrow">↓</div>
          <div class="arch-node">Score Tile Buffer & V-Row Assembler</div>
          <div class="arch-arrow">↓</div>
          <div class="arch-node highlight">AXI4-Stream Egress (Output Row Collector)</div>
        </div>

        <h3>3. Verification Strategy</h3>
        <p>Verified using a multi-tiered verification framework:</p>
        <ul>
          <li><strong>Python Golden Reference Model:</strong> Simulates bit-exact and ULP-tolerant mathematical results for randomized 65,536 BF16 vectors.</li>
          <li><strong>SystemVerilog Scoreboard:</strong> Captures AXI-Stream egress beats, compares against golden mem files, and reports pass/fail with max 64 ULP allowance.</li>
          <li><strong>Boundary Regressions:</strong> Automated python regression runner covering sequence lengths 1, 15, 16, 17, 30, and 32.</li>
        </ul>

        <h3>4. Verified Implementation & Timing Results</h3>
        <div class="timing-metrics-grid" style="margin-top: 1rem;">
          <div class="t-metric positive"><span class="t-label">WNS</span><span class="t-val">+0.204 ns</span></div>
          <div class="t-metric positive"><span class="t-label">WHS</span><span class="t-val">+0.009 ns</span></div>
          <div class="t-metric clean"><span class="t-label">FAILING PATHS</span><span class="t-val">0</span></div>
        </div>
      </div>
    `,
    can: `
      <div class="project-modal-header">
        <div class="project-badge-strip">
          <span class="project-tags">CAN 2.0A/B & CAN FD • CDC • SPARTAN-7</span>
        </div>
        <h2>CAN FD Controller Core</h2>
        <p class="modal-sub">Multi-Module Verilog RTL Controller & CDC Pipeline</p>
      </div>

      <div class="modal-body-content">
        <h3>1. Subsystem Architecture</h3>
        <p>Built across 12 modular Verilog RTL files integrating complete automotive CAN 2.0A/2.0B and CAN FD protocol state machines.</p>
        
        <h3>2. Key Technical Innovations</h3>
        <ul>
          <li><strong>Dual-Clock Domain Crossing (CDC):</strong> Implemented 2-FF synchronizers for single-bit signals and Gray-code asynchronous FIFOs for dual-clock data buffering between host APB clock and CAN bit clock.</li>
          <li><strong>Fault Confinement FSM:</strong> Hardware tracking of Transmit/Receive Error Counters (TEC/REC) driving Error Active, Error Passive, and Bus-Off states.</li>
          <li><strong>CRC Engine:</strong> On-the-fly CRC-15 / CRC-17 calculation with automatic bit-stuffing and frame delimiting.</li>
        </ul>

        <h3>3. Synthesis Utilization (Spartan-7 Target)</h3>
        <div class="hw-metrics-row" style="margin-top: 1rem;">
          <div class="hw-metric-card"><span class="hw-val">1,123</span><span class="hw-lbl">LUTs</span></div>
          <div class="hw-metric-card"><span class="hw-val">1,552</span><span class="hw-lbl">Flip-Flops</span></div>
          <div class="hw-metric-card"><span class="hw-val">APB</span><span class="hw-lbl">Bus Interface</span></div>
        </div>
      </div>
    `
  };

  document.querySelectorAll('.btn-modal-trigger').forEach(btn => {
    btn.addEventListener('click', () => {
      const pKey = btn.getAttribute('data-project');
      if (projectDetails[pKey]) {
        modalContent.innerHTML = projectDetails[pKey];
        projectModal.classList.add('active');
      }
    });
  });

  if (modalCloseBtn) {
    modalCloseBtn.addEventListener('click', () => {
      projectModal.classList.remove('active');
    });
  }
  if (projectModal) {
    projectModal.addEventListener('click', (e) => {
      if (e.target === projectModal) projectModal.classList.remove('active');
    });
  }


  /* ------------------------------------------------------------------------
     5. ENGINEERING TERMINAL CLI EASTER EGG
     ------------------------------------------------------------------------ */
  const terminalModal = document.getElementById('terminal-modal');
  const terminalTrigger = document.getElementById('terminal-trigger');
  const termCloseDot = document.getElementById('term-close-dot');
  const termInput = document.getElementById('term-input');
  const termHistory = document.getElementById('terminal-history');

  if (terminalTrigger && terminalModal) {
    terminalTrigger.addEventListener('click', () => {
      terminalModal.classList.add('active');
      if (termInput) termInput.focus();
    });
  }

  if (termCloseDot) {
    termCloseDot.addEventListener('click', () => {
      terminalModal.classList.remove('active');
    });
  }

  if (terminalModal) {
    terminalModal.addEventListener('click', (e) => {
      if (e.target === terminalModal) terminalModal.classList.remove('active');
    });
  }

  if (termInput) {
    termInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const cmd = termInput.value.trim().toLowerCase();
        termInput.value = '';

        // Add prompt echo
        termHistory.innerHTML += `<div class="term-line"><span class="term-prompt">prad@silicon-lab:~$</span> ${cmd}</div>`;

        // Handle CLI commands
        if (cmd === 'whoami') {
          termHistory.innerHTML += `<div class="term-line">Pradeep M — Electronics & Communication Engineering Student aspiring to be an RTL Design & Verification Engineer.</div>`;
        } else if (cmd === 'focus') {
          termHistory.innerHTML += `<div class="term-line">Focus Areas: Digital Design | RTL Synthesis | SystemVerilog Verification | FPGA Implementation | CDC Architecture</div>`;
        } else if (cmd === 'current_stack') {
          termHistory.innerHTML += `<div class="term-line">Verilog HDL, SystemVerilog, Xilinx Vivado, ModelSim, Synopsys VCS, Cadence NCLaunch/Genus/Innovus, Spartan-7, PYNQ-Z2, Zynq-7000, AXI4-Lite/Stream, Embedded C.</div>`;
        } else if (cmd === 'projects') {
          termHistory.innerHTML += `<div class="term-line">1. CAN FD Controller Core (12 Verilog RTL blocks, CDC, APB)<br>2. BF16 Grouped-Query Attention Hardware Accelerator (32 Q / 8 KV, Llama-3 geometry, Zynq-7000)</div>`;
        } else if (cmd === 'clear') {
          termHistory.innerHTML = `<div class="term-line welcome">Pradeep M Silicon Lab CLI v2.4</div>`;
        } else if (cmd === 'help') {
          termHistory.innerHTML += `<div class="term-line">Commands: <span class="term-cmd">whoami</span>, <span class="term-cmd">focus</span>, <span class="term-cmd">current_stack</span>, <span class="term-cmd">projects</span>, <span class="term-cmd">clear</span></div>`;
        } else if (cmd !== '') {
          termHistory.innerHTML += `<div class="term-line">Command not recognized: '${cmd}'. Type <span class="term-cmd">help</span> for commands.</div>`;
        }

        termHistory.scrollTop = termHistory.scrollHeight;
      }
    });
  }


  /* ------------------------------------------------------------------------
     6. MOBILE MENU TOGGLE
     ------------------------------------------------------------------------ */
  const mobileToggle = document.getElementById('mobile-toggle');
  const navLinks = document.getElementById('nav-links');

  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      navLinks.classList.toggle('active');
    });
  }

});
