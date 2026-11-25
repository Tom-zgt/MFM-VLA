/* MFM-VLA project page interactions (GenFirst style):
   reveal-on-scroll, synchronized side-by-side video gallery, 7-step pipeline inspector,
   figure zoom lightbox, full-reel modal, BibTeX copy, and back-to-top. */

(function () {
  'use strict';

  /* ---------- 1. Reveal on scroll ---------- */
  var revealables = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: '0px 0px -6% 0px', threshold: 0.04 });
    revealables.forEach(function (el) { io.observe(el); });
  } else {
    revealables.forEach(function (el) { el.classList.add('is-in'); });
  }

  /* ---------- 2. Interactive 7-Stage Pipeline Inspector ---------- */
  var WALKTHROUGH_STEPS = [
    {
      idx: '01',
      short: 'Perturbed Scene',
      actor: 'Capability Router',
      roleClass: 'is-mfm',
      img: 'static/walkthrough/step1_init.jpg',
      title: 'Stage 1 · Task-Perturbed Scene & Frozen Suite Routing',
      desc: 'In LIBERO-Object Task Perturbation (init 18), both the memorized training object (red ketchup) and the newly instructed object (milk carton) appear together. The development-calibrated router activates Assisted mode.',
      dgc: '<span class="k">instruction:</span>     "pick the milk and place it in the basket"\n' +
           '<span class="k">router_mode:</span>     <span class="v-hl">ASSISTED</span> (frozen suite: libero_object_task)\n' +
           '<span class="k">shortcut_audit:</span>  Base \u03c00.5 lifts red ketchup in >90% of trials'
    },
    {
      idx: '02',
      short: 'Fixed Grounding',
      actor: 'GPT-6 Astra',
      roleClass: 'is-mfm',
      img: 'static/walkthrough/step2_ground_milk.jpg',
      title: 'Stage 2 · Coarse-to-Fine 2D Manipulation-Point Grounding',
      desc: 'GPT-6 Astra inspects the fixed RGB view (full image followed by zoomed crop) to localize task-specific manipulation points on the milk carton top [380, 265], base contact, and woven basket cavity [479, 286].',
      dgc: '<span class="k">grasp_target:</span>    "milk carton"  <span class="v-hl">fixed_top_uv = [380, 265]</span>\n' +
           '<span class="k">base_contact:</span>    fixed_base_uv = [383, 548]\n' +
           '<span class="k">place_target:</span>    "woven basket" <span class="v-hl">fixed_rim_uv = [479, 286]</span>'
    },
    {
      idx: '03',
      short: '2-View Triangulation',
      actor: 'GPT-6 Astra',
      roleClass: 'is-mfm',
      img: 'static/walkthrough/step3_wrist_survey.jpg',
      title: 'Stage 3 · Active Wrist Survey & Depth-Free 3D Ray Triangulation',
      desc: 'A single free-space wrist-camera survey captures a second calibrated RGB view. Matching semantic points across views triangulates 3D coordinates with a 0.03 cm ray-intersection residual—without depth sensors.',
      dgc: '<span class="k">wrist_survey_uv:</span> [746, 727]\n' +
           '<span class="k">hat_p_grasp:</span>     <span class="v-hl">[-0.188, -0.079, +0.123] m</span>\n' +
           '<span class="k">ray_gap:</span>         <span class="v-hl">0.03 cm</span> (\u2264 2.5 cm consistency gate \u2192 pass)'
    },
    {
      idx: '04',
      short: 'Pre-Grasp Handoff',
      actor: 'Controller \u2192 VLA',
      roleClass: 'is-mfm',
      img: 'static/walkthrough/step4_pre_grasp.jpg',
      title: 'Stage 4 · Free-Space Reach to Pre-Grasp Handoff Pose',
      desc: 'A Cartesian controller moves the open gripper to the demonstration-profiled handoff offset [+2.6, +0.4, +9.0] cm above the milk carton (t = 36), placing the state inside the adapted VLA\'s local attraction basin.',
      dgc: '<span class="k">scene:</span>           milk carton (back-left), woven basket (front-right)\n' +
           '<span class="k">subgoal:</span>         lower gripper straight down to milk carton\n' +
           '<span class="k">geometry:</span>        target is <span class="v-hl">2 cm back, 9 cm below</span> end-effector\n' +
           '<span class="k">next 10 steps:</span>   move 5 cm down; close gripper to grasp'
    },
    {
      idx: '05',
      short: 'VLA Grasp & Lift',
      actor: 'Adapted \u03c00.5',
      roleClass: 'is-vla',
      img: 'static/walkthrough/step5_vla_grasp.jpg',
      title: 'Stage 5 · Closed-Loop VLA Descent, Contact Alignment & Lift',
      desc: 'Conditioned on action-chunk Dense Geometric Captions, Adapted \u03c00.5 performs fine-grained finger alignment, closes the gripper around the carton (t = 55), and lifts smoothly (t = 65).',
      dgc: '<span class="k">scene:</span>           milk carton (back-left), woven basket (front-right)\n' +
           '<span class="k">subgoal:</span>         lift grasped milk carton clear of tabletop\n' +
           '<span class="k">geometry:</span>        destination basket is <span class="v-hl">38 cm front-right, 4 cm above</span>\n' +
           '<span class="k">next 10 steps:</span>   move 6 cm up; keep gripper closed'
    },
    {
      idx: '06',
      short: 'Verify & Carry',
      actor: 'Monitor + Watch-Carry',
      roleClass: 'is-mfm',
      img: 'static/walkthrough/step6_pre_place.jpg',
      title: 'Stage 6 · Grasp Verification & Watch-and-Takeover Carry',
      desc: 'The grasp monitor verifies that the milk carton (not a distractor) is securely held. Watch-and-Takeover monitors horizontal progress toward the basket and guides transport to the pre-place handoff pose (t = 111).',
      dgc: '<span class="k">check_grasp:</span>     <span class="v-hl">held = "milk carton", verified = true</span>\n' +
           '<span class="k">carry_monitor:</span>   Watch-and-Takeover (W0=20, \u0394d\u22653 cm, T_max=40)\n' +
           '<span class="k">pre_place_3d:</span>    <span class="v-hl">[-0.008, +0.272, +0.281] m</span>'
    },
    {
      idx: '07',
      short: 'VLA Place & Release',
      actor: 'Adapted \u03c00.5',
      roleClass: 'is-vla',
      img: 'static/walkthrough/step7_placed.jpg',
      title: 'Stage 7 · Closed-Loop VLA Lowering & Clean Release',
      desc: 'Control returns to Adapted \u03c00.5 above the basket opening; the policy lowers the milk carton into the basket cavity, opens the gripper (t = 138), and completes the episode cleanly (t = 153 / 300).',
      dgc: '<span class="k">episode_result:</span>  <span class="v-hl">SUCCESS (153 / 300 steps)</span>\n' +
           '<span class="k">mfm_queries:</span>     3 sparse calls (ground + wrist match + verify)\n' +
           '<span class="k">contact_control:</span> 100% closed-loop Adapted \u03c00.5'
    }
  ];

  var stepPillsEl = document.getElementById('stepPills');
  function selectStep(i) {
    var st = WALKTHROUGH_STEPS[i];
    if (!st) return;
    document.querySelectorAll('.step-pill').forEach(function (b, idx) {
      b.classList.toggle('is-on', idx === i);
    });
    var img = document.getElementById('walkImg');
    var badge = document.getElementById('walkBadge');
    if (img) {
      img.src = st.img;
      img.dataset.caption = st.title + ' — ' + st.desc;
    }
    if (badge) {
      badge.textContent = st.actor;
      badge.className = 'actor-badge ' + st.roleClass;
    }
    var titleEl = document.getElementById('walkTitle');
    var descEl = document.getElementById('walkDesc');
    var dgcEl = document.getElementById('walkDgc');
    if (titleEl) titleEl.textContent = st.title;
    if (descEl) descEl.textContent = st.desc;
    if (dgcEl) dgcEl.innerHTML = st.dgc;
  }

  if (stepPillsEl) {
    WALKTHROUGH_STEPS.forEach(function (st, i) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'step-pill' + (i === 0 ? ' is-on' : '');
      btn.innerHTML = '<span class="s-idx">' + st.idx + '</span>' + st.short;
      btn.addEventListener('click', function () { selectStep(i); });
      stepPillsEl.appendChild(btn);
    });
    selectStep(0);
  }

  /* ---------- 3. Curated Synchronized Paired Video Gallery ---------- */
  var DEMO_PAIRS = [
    {
      cat: 'libero',
      featured: true,
      square: true,
      bench: 'LIBERO-PRO',
      tag: 'Task Perturbation',
      instruction: 'Pick the milk and place it in the basket.',
      leftSrc: 'static/videos/libero_1_milk_left.mp4',
      rightSrc: 'static/videos/libero_1_milk_right.mp4',
      leftLabel: '\u03c0<sub>0.5</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0.5</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Visual scene shortcut: ignores updated target ("milk") and grasps the memorized red ketchup bottle.',
      passNote: 'Two-view RGB triangulation grounds the milk carton and basket rim; Adapted \u03c00.5 grasps and places cleanly.'
    },
    {
      cat: 'libero',
      featured: true,
      square: true,
      bench: 'LIBERO-PRO',
      tag: 'Position Perturbation',
      instruction: 'Open the middle layer of the drawer.',
      leftSrc: 'static/videos/libero_2_drawer_left.mp4',
      rightSrc: 'static/videos/libero_2_drawer_right.mp4',
      leftLabel: '\u03c0<sub>0.5</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0.5</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Reaches toward the unperturbed training cabinet pose and misses the displaced drawer handle.',
      passNote: 'Localizes the displaced middle handle in 3D and hands off inside \u03c00.5\'s articulated pulling basin.'
    },
    {
      cat: 'libero',
      featured: false,
      square: true,
      bench: 'LIBERO-PRO',
      tag: 'Position Perturbation (Object Swap)',
      instruction: 'Pick up the orange juice and place it in the basket.',
      leftSrc: 'static/videos/libero_3_oj_left.mp4',
      rightSrc: 'static/videos/libero_3_oj_right.mp4',
      leftLabel: '\u03c0<sub>0.5</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0.5</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Executes memorized slot trajectory and lifts the wrong carton (chocolate pudding).',
      passNote: 'Grounds the swapped orange juice carton, verifies grasp identity, and places into the basket.'
    },
    {
      cat: 'aloha',
      featured: true,
      square: false,
      bench: 'ALOHA-Sim-PRO',
      tag: 'Position Perturbation (+8 cm OOD)',
      instruction: 'Transfer cube (14-DoF bimanual handover).',
      leftSrc: 'static/videos/aloha_1_8cm_left.mp4',
      rightSrc: 'static/videos/aloha_1_8cm_right.mp4',
      leftLabel: '\u03c0<sub>0</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Right arm sweeps empty center workspace and misses the +8 cm out-of-distribution cube.',
      passNote: 'Grounds the OOD cube and routes it into the canonical mid-air corridor where \u03c00 completes bimanual handover.'
    },
    {
      cat: 'aloha',
      featured: false,
      square: false,
      bench: 'ALOHA-Sim-PRO',
      tag: 'Position Perturbation (+12 cm OOD)',
      instruction: 'Transfer cube (extreme +12 cm workspace displacement).',
      leftSrc: 'static/videos/aloha_2_12cm_left.mp4',
      rightSrc: 'static/videos/aloha_2_12cm_right.mp4',
      leftLabel: '\u03c0<sub>0</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Closes gripper in empty air near training mean; bimanual coordination stalls.',
      passNote: 'Acquires the far-displaced cube and restores smooth dual-arm coordination at the handover zone.'
    },
    {
      cat: 'aloha',
      featured: false,
      square: false,
      bench: 'ALOHA-Sim-PRO',
      tag: 'Object Perturbation (Unseen Color)',
      instruction: 'Transfer the blue cube.',
      leftSrc: 'static/videos/aloha_3_blue_left.mp4',
      rightSrc: 'static/videos/aloha_3_blue_right.mp4',
      leftLabel: '\u03c0<sub>0</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Unseen object color shifts visual features, causing unstable approach and missed grasp.',
      passNote: 'Open-vocabulary MFM grounding is invariant to unseen color and hands off cleanly for transfer.'
    },
    {
      cat: 'aloha',
      featured: true,
      square: false,
      bench: 'ALOHA-Sim-PRO',
      tag: 'Task + Color + Position Perturbation',
      instruction: 'Ignore the default center position; pick up the green cube and transfer it to the left arm.',
      leftSrc: 'static/videos/aloha_4_green_left.mp4',
      rightSrc: 'static/videos/aloha_4_green_right.mp4',
      leftLabel: '\u03c0<sub>0</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Ignores language target specification and reaches toward the default distractor cube.',
      passNote: 'Disambiguates the instructed green cube among distractors and completes bimanual transfer (100% Task SR).'
    },
    {
      cat: 'robocasa',
      featured: true,
      square: false,
      bench: 'RoboCasa365',
      tag: 'Held-Out Unseen Kitchen',
      instruction: 'Place the toasted item on a plate.',
      leftSrc: 'static/videos/rc_1_toast_left.mp4',
      rightSrc: 'static/videos/rc_1_toast_right.mp4',
      leftLabel: '\u03c0<sub>0.5</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0.5</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Drifts in unseen kitchen layout and fails to extract toast cleanly onto the plate.',
      passNote: 'Adapted \u03c00.5 + calibrated routing extracts the toasted bread and places it onto the target plate.'
    },
    {
      cat: 'robocasa',
      featured: false,
      square: false,
      bench: 'RoboCasa365',
      tag: 'Held-Out Unseen Kitchen',
      instruction: 'Pick the mug from the counter and place it under the coffee machine dispenser.',
      leftSrc: 'static/videos/rc_3_mug_left.mp4',
      rightSrc: 'static/videos/rc_3_mug_right.mp4',
      leftLabel: '\u03c0<sub>0.5</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0.5</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Misaligns mug placement relative to the recessed coffee machine drip tray.',
      passNote: 'Aligns mug beneath the dispenser nozzle in unseen countertop geometry.'
    },
    {
      cat: 'robocasa',
      featured: false,
      square: false,
      bench: 'RoboCasa365',
      tag: 'Held-Out Unseen Kitchen (Assisted)',
      instruction: 'Close the toaster oven door.',
      leftSrc: 'static/videos/rc_4_oven_left.mp4',
      rightSrc: 'static/videos/rc_4_oven_right.mp4',
      leftLabel: '\u03c0<sub>0.5</sub> (Baseline)',
      rightLabel: '\u03c0<sub>0.5</sub> + MFM-VLA (Ours)',
      leftStatus: 'Fail',
      rightStatus: 'Success',
      failNote: 'Misses the articulated toaster-oven door handle and leaves the door open.',
      passNote: 'Frozen task router invokes MFM grounding on the oven door handle for complete articulated closure.'
    },
    {
      cat: 'realworld',
      featured: true,
      square: false,
      bench: 'Franka–XHand Real-World',
      tag: 'Multi-Stage Scene Generalization',
      instruction: 'Pick the transparent glass funnel from the stand and insert it into the round-bottom flask.',
      leftSrc: 'static/videos/rw_3_mid_scene_left.mp4',
      rightSrc: 'static/videos/rw_3_mid_scene_right.mp4',
      leftLabel: 'MFM-VLA (Ours) · Front-Left RGB',
      rightLabel: 'MFM-VLA (Ours) · Overhead RGB',
      leftStatus: 'Sync View 1',
      rightStatus: 'Sync View 2',
      bothOurs: true,
      failNote: 'Depth-free RGB-only execution amidst unseen laboratory distractors (Base \u03c00.5 multi-stage SR: 0.8%).',
      passNote: '12-DoF XHand grasps transparent glass stem from stand and inserts into narrow flask neck (31.7% multi-stage SR).'
    },
    {
      cat: 'realworld',
      featured: false,
      square: false,
      bench: 'Franka–XHand Real-World',
      tag: 'Position Generalization',
      instruction: 'Extract the transparent glass funnel from the displaced round-bottom flask.',
      leftSrc: 'static/videos/rw_1_remove_left.mp4',
      rightSrc: 'static/videos/rw_1_remove_right.mp4',
      leftLabel: 'MFM-VLA (Ours) · Front-Left RGB',
      rightLabel: 'MFM-VLA (Ours) · Overhead RGB',
      leftStatus: 'Sync View 1',
      rightStatus: 'Sync View 2',
      bothOurs: true,
      failNote: 'Base \u03c00.5 achieves 0.0% on Remove Funnel under position shift due to refractive specularities and unseen pose.',
      passNote: 'Calibrated RGB triangulation localizes the displaced transparent funnel (85.0% Position SR vs. 0.0% base).'
    },
    {
      cat: 'realworld',
      featured: false,
      square: false,
      bench: 'Franka–XHand Real-World',
      tag: 'Scene Generalization',
      instruction: 'Grasp the transparent glass funnel from the acrylic stand amidst unseen distractors.',
      leftSrc: 'static/videos/rw_2_pick_left.mp4',
      rightSrc: 'static/videos/rw_2_pick_right.mp4',
      leftLabel: 'MFM-VLA (Ours) · Front-Left RGB',
      rightLabel: 'MFM-VLA (Ours) · Overhead RGB',
      leftStatus: 'Sync View 1',
      rightStatus: 'Sync View 2',
      bothOurs: true,
      failNote: 'Cluttered chemistry bench distractors cause standalone VLA policies to drift toward wrong vessels (15.0% SR).',
      passNote: 'MFM grounds the correct funnel amidst clutter and guides XHand pre-grasp alignment (80.0% Scene SR).'
    },
    {
      cat: 'realworld',
      featured: false,
      square: false,
      bench: 'Franka–XHand Real-World',
      tag: 'Multi-Stage Position Generalization',
      instruction: 'Pick the glass funnel from the stand and insert it into the displaced round-bottom flask.',
      leftSrc: 'static/videos/rw_4_mid_pos_left.mp4',
      rightSrc: 'static/videos/rw_4_mid_pos_right.mp4',
      leftLabel: 'MFM-VLA (Ours) · Front-Left RGB',
      rightLabel: 'MFM-VLA (Ours) · Overhead RGB',
      leftStatus: 'Sync View 1',
      rightStatus: 'Sync View 2',
      bothOurs: true,
      failNote: 'Requires chaining precision transparent-stem grasping with sub-centimeter insertion at an unseen flask pose.',
      passNote: 'Two-stage MFM spatial guidance + closed-loop XHand control completes full pick-and-insert trajectory.'
    }
  ];

  var videoObserver = null;

  function bindSyncPair(block) {
    var vids = Array.prototype.slice.call(block.querySelectorAll('video'));
    if (vids.length !== 2) return;
    var vL = vids[0], vR = vids[1];
    var suppressPlay = new WeakSet();
    var suppressPause = new WeakSet();
    var suppressSeek = new WeakSet();

    function markTemp(set, el, ms) {
      set.add(el);
      setTimeout(function () { set.delete(el); }, ms || 180);
    }
    function isAtEnd(v) {
      return v.ended || (Number.isFinite(v.duration) && v.duration > 0 && v.currentTime >= v.duration - 0.35);
    }
    function wire(src, dst) {
      src.addEventListener('play', function () {
        if (suppressPlay.has(src)) { suppressPlay.delete(src); return; }
        if (isAtEnd(src) || (src.currentTime < 0.3 && isAtEnd(dst))) {
          if (src.currentTime > 0.05) { markTemp(suppressSeek, src); src.currentTime = 0; }
          markTemp(suppressSeek, dst);
          dst.currentTime = 0;
        } else if (Math.abs(src.currentTime - dst.currentTime) > 0.2) {
          markTemp(suppressSeek, dst);
          dst.currentTime = src.currentTime;
        }
        if (dst.paused) {
          markTemp(suppressPlay, dst);
          dst.play().catch(function () {});
        }
      });
      src.addEventListener('pause', function () {
        if (suppressPause.has(src)) { suppressPause.delete(src); return; }
        if (isAtEnd(src)) return;
        if (!dst.paused && !isAtEnd(dst)) {
          markTemp(suppressPause, dst);
          dst.pause();
        }
      });
      src.addEventListener('seeking', function () {
        if (suppressSeek.has(src)) { suppressSeek.delete(src); return; }
        if (Math.abs(src.currentTime - dst.currentTime) > 0.08) {
          markTemp(suppressSeek, dst);
          dst.currentTime = src.currentTime;
        }
      });
    }
    wire(vL, vR);
    wire(vR, vL);
  }

  function renderDemos(filter) {
    var list = document.getElementById('demoList');
    if (!list) return;
    if (videoObserver) videoObserver.disconnect();
    list.innerHTML = '';

    var items = DEMO_PAIRS.filter(function (d) {
      if (filter === 'featured') return d.featured;
      if (filter === 'all') return true;
      return d.cat === filter;
    });

    items.forEach(function (d) {
      var card = document.createElement('div');
      card.className = 'demo-pair-card';
      var sq = d.square ? ' is-square' : '';
      var leftOurs = d.bothOurs ? ' is-ours' : '';
      var leftStatusCls = d.bothOurs ? 'is-pass' : 'is-fail';
      card.innerHTML =
        '<div class="demo-pair-head">' +
          '<div class="demo-instruction"><span class="kw">Task</span>“' + d.instruction + '”</div>' +
          '<div class="demo-badges">' +
            '<span class="meta-pill">' + d.bench + '</span>' +
            '<span class="meta-pill is-accent">' + d.tag + '</span>' +
          '</div>' +
        '</div>' +
        '<div class="video-pair-grid">' +
          '<div class="v-panel' + sq + leftOurs + '">' +
            '<video src="' + d.leftSrc + '" controls muted playsinline preload="metadata"></video>' +
            '<div class="v-caption">' +
              '<span>' + d.leftLabel + '</span>' +
              '<span class="v-status ' + leftStatusCls + '">' + d.leftStatus + '</span>' +
            '</div>' +
          '</div>' +
          '<div class="v-panel is-ours' + sq + '">' +
            '<video src="' + d.rightSrc + '" controls muted playsinline preload="metadata"></video>' +
            '<div class="v-caption">' +
              '<span>' + d.rightLabel + '</span>' +
              '<span class="v-status is-pass">' + d.rightStatus + '</span>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="demo-notes">' +
          '<div><b class="' + (d.bothOurs ? 'good' : 'bad') + '">' + (d.bothOurs ? 'Challenge:' : '\u2715 Baseline:') + '</b> ' + d.failNote + '</div>' +
          '<div><b class="good">\u2713 MFM-VLA:</b> ' + d.passNote + '</div>' +
        '</div>';
      bindSyncPair(card);
      list.appendChild(card);
    });

    if ('IntersectionObserver' in window) {
      videoObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          var vids = Array.prototype.slice.call(entry.target.querySelectorAll('video'));
          if (entry.isIntersecting) {
            var allEnded = vids.length > 0 && vids.every(function (v) {
              return v.ended || (v.duration && v.currentTime >= v.duration - 0.35);
            });
            if (!allEnded) {
              vids.forEach(function (v) {
                if (v.paused && !v.ended) v.play().catch(function () {});
              });
            }
          } else {
            vids.forEach(function (v) { if (!v.paused) v.pause(); });
          }
        });
      }, { threshold: 0.35 });
      list.querySelectorAll('.demo-pair-card').forEach(function (c) { videoObserver.observe(c); });
    }
  }

  document.querySelectorAll('#demoSwitch button').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('#demoSwitch button').forEach(function (b) {
        b.classList.toggle('is-on', b === btn);
      });
      renderDemos(btn.dataset.filter);
    });
  });
  renderDemos('featured');

  /* ---------- 4. Full 4-Min Demo Reel Modal ---------- */
  var vModal = document.getElementById('videoModal');
  var vPlayer = document.getElementById('reelPlayer');
  function openReel() {
    if (!vModal || !vPlayer) return;
    vPlayer.src = 'static/videos/00_ALL_BENCHMARKS_SUPPLEMENTARY_DEMO.mp4';
    vModal.classList.add('is-open');
    vModal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    vPlayer.play().catch(function () {});
  }
  function closeReel() {
    if (!vModal || !vPlayer) return;
    vPlayer.pause();
    vPlayer.src = '';
    vModal.classList.remove('is-open');
    vModal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  }
  document.querySelectorAll('.open-reel-trigger').forEach(function (el) {
    el.addEventListener('click', openReel);
  });
  var vClose = document.getElementById('closeReelBtn');
  if (vClose) vClose.addEventListener('click', closeReel);
  if (vModal) {
    vModal.addEventListener('click', function (ev) {
      if (ev.target === vModal) closeReel();
    });
  }

  /* ---------- 5. Image Lightbox ---------- */
  var lb = document.querySelector('.lightbox');
  var lbImg = lb ? lb.querySelector('img') : null;
  var lbCap = lb ? lb.querySelector('.lb-caption') : null;

  function closeLightbox() {
    if (!lb) return;
    lb.classList.remove('is-open');
    lb.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  }
  if (lb) {
    document.addEventListener('click', function (ev) {
      var fig = ev.target.closest ? ev.target.closest('.zoomable') : null;
      if (fig && fig.tagName === 'IMG') {
        lbImg.src = fig.src;
        lbCap.textContent = fig.dataset.caption || fig.alt || '';
        lb.classList.add('is-open');
        lb.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
      }
    });
    lb.querySelector('.lb-close').addEventListener('click', closeLightbox);
    lb.addEventListener('click', function (ev) {
      if (ev.target === lb || ev.target === lbImg) closeLightbox();
    });
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') {
        closeLightbox();
        closeReel();
      }
    });
  }

  /* ---------- 6. BibTeX Copy ---------- */
  var copyBtn = document.querySelector('.copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var text = document.querySelector('.bibtex-block pre').innerText;
      var done = function () {
        copyBtn.textContent = 'Copied';
        setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1600);
      };
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        done();
      }
    });
  }

  /* ---------- 7. Back to Top ---------- */
  var topBtn = document.querySelector('.back-to-top');
  if (topBtn) {
    window.addEventListener('scroll', function () {
      topBtn.classList.toggle('is-shown', window.scrollY > 700);
    }, { passive: true });
    topBtn.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }
})();
