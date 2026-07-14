// ---------- derived fuel consumption ----------
  const cityEl = document.getElementById('city');
  const hwyEl = document.getElementById('hwy');
  const combValEl = document.getElementById('comb-val');
  const mpgValEl = document.getElementById('mpg-val');

  function updateDerived(){
    const city = parseFloat(cityEl.value);
    const hwy = parseFloat(hwyEl.value);
    if(!isNaN(city) && !isNaN(hwy)){
      const comb = city*0.55 + hwy*0.45;
      combValEl.textContent = comb.toFixed(1);
      mpgValEl.textContent = Math.round(235.21 / comb);
      return { comb, mpg: 235.21/comb };
    } else {
      combValEl.textContent = '—';
      mpgValEl.textContent = '—';
      return null;
    }
  }
  cityEl.addEventListener('input', updateDerived);
  hwyEl.addEventListener('input', updateDerived);

  // ---------- engine slider ----------
  const engineEl = document.getElementById('engine');
  const engineValEl = document.getElementById('engine-val');
  engineEl.addEventListener('input', () => {
    engineValEl.textContent = parseFloat(engineEl.value).toFixed(1) + ' L';
  });

  // ---------- cylinder stepper ----------
  const cylEl = document.getElementById('cyl');
  document.getElementById('cyl-plus').addEventListener('click', () => {
    cylEl.value = Math.min(16, parseInt(cylEl.value||4) + 1);
  });
  document.getElementById('cyl-minus').addEventListener('click', () => {
    cylEl.value = Math.max(2, parseInt(cylEl.value||4) - 1);
  });

  // ---------- make -> model cascading, searchable dropdown ----------
  // MODEL_DATA maps each raw "Make" value (exactly as in the dataset) to the
  // list of raw "Model" values for that make (also exactly as in the dataset).
  // Loaded from static/models_by_make.json, which was generated from the
  // training CSV so the values sent to Flask always match the dataset.
  let MODEL_DATA = {};

  const makeEl = document.getElementById('make');
  const modelEl = document.getElementById('model');           // visible, user-facing text input
  const modelDatalist = document.getElementById('model-options');
  const modelRawEl = document.getElementById('model-raw');    // hidden input, name="model" -> posted to Flask

  // Display rule: only the first letter is capitalized, the rest of the
  // dataset string is left untouched so acronyms (e.g. "MDX SH-AWD") stay readable.
  function toDisplayModel(raw){
    if(!raw) return raw;
    return raw.charAt(0).toUpperCase() + raw.slice(1);
  }

    fetch('/static/models_by_make.json')   
    .then(res => res.json())
    .then(data => { MODEL_DATA = data; })
    .catch(() => { MODEL_DATA = {}; });

  function populateModelsForMake(){
    const make = makeEl.value;
    const models = MODEL_DATA[make] || [];

    modelDatalist.innerHTML = '';
    modelEl.value = '';
    modelRawEl.value = '';

    if(models.length){
      modelEl.disabled = false;
      modelEl.placeholder = 'Type to search a model…';
      models.forEach(raw => {
        const opt = document.createElement('option');
        opt.value = toDisplayModel(raw);
        modelDatalist.appendChild(opt);
      });
    } else {
      modelEl.disabled = true;
      modelEl.placeholder = 'Select a make first';
    }
  }

  // Resolves whatever the user typed/selected back to the exact raw dataset value.
  function resolveRawModel(typed){
    const make = makeEl.value;
    const models = MODEL_DATA[make] || [];
    const needle = typed.trim().toLowerCase();
    if(!needle) return '';
    for(const raw of models){
      if(toDisplayModel(raw).toLowerCase() === needle || raw.toLowerCase() === needle){
        return raw;
      }
    }
    return '';
  }

  makeEl.addEventListener('change', populateModelsForMake);

  modelEl.addEventListener('input', () => {
    modelRawEl.value = resolveRawModel(modelEl.value);
  });

  // ---------- prediction ----------
  const FUEL_LABEL = { X:'Regular gasoline', Z:'Premium gasoline', D:'Diesel', E:'Ethanol (E85)', N:'Natural gas' };

  const form = document.getElementById('predict-form');
  const msgEl = document.getElementById('form-msg');
  const resultCar = document.getElementById('result-car');
  const miniMarker = document.getElementById('mini-marker');
  const co2NumEl = document.getElementById('co2-num');
  const ratingBadge = document.getElementById('rating-badge');
  const resultSide = document.getElementById('result-side');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msgEl.textContent = '';

    const make = makeEl.value;
    const modelRaw = modelRawEl.value;
    const modelDisplay = modelEl.value.trim();

    // .value on these selects is the exact dataset code (e.g. "A4", "COMPACT");
    // the option's visible text (e.g. "Automatic — 4 speed (A4)") is only for display.
    const vclassEl = document.getElementById('vclass');
    const transEl = document.getElementById('trans');
    const vclass = vclassEl.value;
    const trans = transEl.value;
    const vclassDisplay = vclassEl.options[vclassEl.selectedIndex]?.text || vclass;
    const transDisplay = transEl.options[transEl.selectedIndex]?.text || trans;

    const engine = parseFloat(engineEl.value);
    const cyl = parseInt(cylEl.value);
    const fuel = document.querySelector('input[name="fuel"]:checked').value;
    const derived = updateDerived();

    if(!make || !modelRaw || !vclass || !trans || !derived){
      msgEl.textContent = !make || !modelDisplay
        ? 'Fill in every field — the model needs the full picture.'
        : !modelRaw
        ? 'Pick a model from the dropdown for the selected make.'
        : 'Fill in every field — the model needs the full picture.';
      return;
    }

    const comb = derived.comb;

    msgEl.textContent = 'Running the model…';

    let co2Rounded;
    try{
      const response = await fetch(form.getAttribute('action') || '/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          make,
          model: modelRaw,
          vclass,
          engine,
          cyl,
          trans,
          fuel,
          city: parseFloat(cityEl.value),
          hwy: parseFloat(hwyEl.value),
          comb
        })
      });

      if(!response.ok) throw new Error('Request failed');
      const data = await response.json();

      // Accepts whichever key the Flask app returns the prediction under.
      const predicted = data.co2 ?? data.prediction ?? data.result ?? data.output ?? data.value;
      if(predicted === undefined || predicted === null || isNaN(predicted)){
        throw new Error('No prediction in response');
      }
      co2Rounded = Math.round(Number(predicted));
    } catch(err){
      msgEl.textContent = "Couldn't get a prediction from the model — check the Flask endpoint and try again.";
      return;
    }

    msgEl.textContent = '';

    const clamped = Math.max(0, Math.min(500, co2Rounded));

    let level;
    if(co2Rounded < 150) level = 'low';
    else if(co2Rounded < 250) level = 'mid';
    else level = 'high';

    // ---------- environmental impact, based on a typical 20,000 km driving year ----------
    const ANNUAL_KM = 20000;
    const KG_CO2_PER_TREE_YEAR = 21; // commonly cited rough absorption rate for one mature tree
    const kgPerYear = (co2Rounded * ANNUAL_KM) / 1000;
    const treesPerYear = Math.max(1, Math.round(kgPerYear / KG_CO2_PER_TREE_YEAR));
    const litersPerYear = Math.round((comb * ANNUAL_KM) / 100);
    const fiveYearTonnes = (kgPerYear * 5 / 1000);

    const impactNote = level === 'low'
      ? `That's a light footprint — this sits toward the cleaner end of what we see across the training data.`
      : level === 'mid'
      ? `That's a moderate footprint. Steady speeds, a lighter foot on the pedal, and regular maintenance can trim it further.`
      : `That's a heavy footprint. Across a few years of driving, the gap between this and a lower-emission vehicle adds up to real tonnes of CO₂.`;

    // the exhaust visual is the payoff: smoke color/density shifts with the level
    resultCar.setAttribute('data-level', level);
    resultCar.classList.remove('rev');
    void resultCar.offsetWidth; // restart the rev animation
    resultCar.classList.add('rev');

    miniMarker.style.left = (clamped/500*100) + '%';

    co2NumEl.textContent = co2Rounded;

    ratingBadge.style.visibility = 'visible';
    ratingBadge.classList.remove('low','mid','high');
    const ratingText = level === 'low' ? 'Low emissions' : level === 'mid' ? 'Moderate emissions' : 'High emissions';
    ratingBadge.textContent = ratingText;
    ratingBadge.classList.add(level);

    resultSide.innerHTML = `
      <h3>${toDisplayModel(make)} ${modelDisplay}</h3>
      <ul class="factor-list">
        <li><span>Vehicle class</span><b>${vclassDisplay}</b></li>
        <li><span>Engine</span><b>${engine.toFixed(1)} L · ${cyl} cyl</b></li>
        <li><span>Transmission</span><b>${transDisplay}</b></li>
        <li><span>Fuel type</span><b>${FUEL_LABEL[fuel]}</b></li>
        <li><span>Combined consumption</span><b>${comb.toFixed(1)} L/100km · ${Math.round(derived.mpg)} mpg</b></li>
      </ul>
      <p class="method-note">This is the CO₂ figure returned by the trained model for these specs.</p>

      <div class="impact-block">
        <div class="impact-head">Environmental impact <span class="hint">— assuming ${ANNUAL_KM.toLocaleString()} km driven per year</span></div>
        <div class="impact-grid">
          <div class="impact-tile ${level}"><b>${Math.round(kgPerYear).toLocaleString()} kg</b><span>CO₂ per year</span></div>
          <div class="impact-tile ${level}"><b>${treesPerYear}</b><span>mature trees to offset it, every year</span></div>
          <div class="impact-tile"><b>${litersPerYear.toLocaleString()} L</b><span>fuel burned per year</span></div>
          <div class="impact-tile ${level}"><b>${fiveYearTonnes.toFixed(1)} t</b><span>CO₂ over 5 years of driving</span></div>
        </div>
        <p class="impact-note ${level}">${impactNote}</p>
      </div>
    `;

    document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  document.getElementById('clear-btn').addEventListener('click', (e) => {
    e.preventDefault();
    form.reset();
    engineValEl.textContent = '2.0 L';
    combValEl.textContent = '—';
    mpgValEl.textContent = '—';
    co2NumEl.textContent = '—';
    ratingBadge.style.visibility = 'hidden';
    resultCar.setAttribute('data-level', 'idle');
    miniMarker.style.left = '0%';
    modelEl.disabled = true;
    modelEl.placeholder = 'Select a make first';
    modelDatalist.innerHTML = '';
    modelRawEl.value = '';
    resultSide.innerHTML = `
      <div class="result-placeholder">
        <h3>Nothing to show yet</h3>
        <p>Fill in the specs above and hit <strong>Run the model</strong> — the exhaust, the number, and the breakdown will fill in right here.</p>
      </div>`;
  });

  // ---------- scroll-driven motion: parallax layers + progress dial + molecule disperse ----------
  const parallaxEls = document.querySelectorAll('[data-parallax]');
  const dial = document.getElementById('scroll-dial');
  const dialLabel = document.getElementById('scroll-dial-label');
  const heroEl = document.getElementById('hero');
  let ticking = false;

  function onScroll(){
    const scrollY = window.scrollY;
    const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
    const pct = maxScroll > 0 ? Math.min(100, Math.max(0, (scrollY / maxScroll) * 100)) : 0;

    parallaxEls.forEach(el => {
      const speed = parseFloat(el.dataset.parallax) || 0.2;
      el.style.setProperty('--py', `${scrollY * speed * -0.35}px`);
    });

    // how far we've scrolled past the hero, 0 -> 1, drives the molecule's disperse
    if(heroEl){
      const heroHeight = heroEl.offsetHeight || window.innerHeight;
      window.moleculeScrollProgress = Math.min(1, Math.max(0, scrollY / heroHeight));
    }

    dial.style.setProperty('--p', pct.toFixed(1));
    dialLabel.textContent = Math.round(pct) + '%';
    dial.classList.toggle('show', scrollY > 80);

    ticking = false;
  }
  window.addEventListener('scroll', () => {
    if(!ticking){ requestAnimationFrame(onScroll); ticking = true; }
  }, { passive:true });
  onScroll();

  // ---------- scroll reveal (fade/slide + staggered groups + line-mask text) ----------
  const revealEls = document.querySelectorAll('.reveal, .form-groups, .result-wrap, .reveal-lines');
  const io = new IntersectionObserver((entries) => {
    entries.forEach(en => { if(en.isIntersecting){ en.target.classList.add('in'); io.unobserve(en.target); } });
  }, { threshold: 0.12 });
  revealEls.forEach(el => io.observe(el));

  // hero headline reveals immediately on load rather than waiting on scroll
  window.addEventListener('DOMContentLoaded', () => {
    const heroLines = document.querySelector('.hero .reveal-lines');
    if(heroLines){ requestAnimationFrame(() => setTimeout(() => heroLines.classList.add('in'), 120)); }

    // count up the trust stats
    const counters = [
      { el: document.getElementById('stat-1'), target: 5 },
      { el: document.getElementById('stat-2'), target: 11 }
    ];
    counters.forEach(({el, target}) => {
      if(!el) return;
      let cur = 0;
      const step = () => {
        cur += Math.max(1, Math.round(target/16));
        if(cur >= target){ el.textContent = target; return; }
        el.textContent = cur;
        requestAnimationFrame(step);
      };
      setTimeout(() => requestAnimationFrame(step), 400);
    });
  });

  // ---------- WebGL CO2 molecule centerpiece ----------
  (function initMolecule(){
    const canvas = document.getElementById('molecule-canvas');
    if(!canvas || typeof THREE === 'undefined') return;
    if(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let renderer;
    try{
      renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
    } catch(e){ return; }

    const container = canvas.parentElement;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/Math.max(container.clientHeight,1), 0.1, 100);
    camera.position.z = 8.5;

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);

    const molecule = new THREE.Group();
    scene.add(molecule);
    const parts = [];

    function randomInSphere(radius){
      const u = Math.random(), v = Math.random();
      const theta = 2*Math.PI*u;
      const phi = Math.acos(2*v-1);
      const r = radius*Math.cbrt(Math.random());
      return [ r*Math.sin(phi)*Math.cos(theta), r*Math.sin(phi)*Math.sin(theta), r*Math.cos(phi) ];
    }

    function makeCluster(count, radius, center, color, size, opacity){
      const positions = new Float32Array(count*3);
      const home = new Float32Array(count*3);
      const dispersed = new Float32Array(count*3);
      for(let i=0;i<count;i++){
        const [x,y,z] = randomInSphere(radius);
        const px = center[0]+x, py = center[1]+y, pz = center[2]+z;
        positions[i*3]=px; positions[i*3+1]=py; positions[i*3+2]=pz;
        home[i*3]=px; home[i*3+1]=py; home[i*3+2]=pz;
        dispersed[i*3] = (Math.random()-0.5)*28;
        dispersed[i*3+1] = (Math.random()-0.5)*28;
        dispersed[i*3+2] = (Math.random()-0.5)*28;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
      const mat = new THREE.PointsMaterial({
        color, size, transparent:true, opacity,
        blending:THREE.AdditiveBlending, depthWrite:false
      });
      const pts = new THREE.Points(geo, mat);
      pts.userData.home = home;
      pts.userData.dispersed = dispersed;
      pts.userData.baseOpacity = opacity;
      molecule.add(pts);
      parts.push(pts);
      return pts;
    }

    function makeBond(from, to, count, color){
      const positions = new Float32Array(count*3);
      const home = new Float32Array(count*3);
      const dispersed = new Float32Array(count*3);
      for(let i=0;i<count;i++){
        const t = Math.random();
        const px = from[0]+(to[0]-from[0])*t + (Math.random()-0.5)*0.08;
        const py = from[1]+(to[1]-from[1])*t + (Math.random()-0.5)*0.08;
        const pz = from[2]+(to[2]-from[2])*t + (Math.random()-0.5)*0.08;
        positions[i*3]=px; positions[i*3+1]=py; positions[i*3+2]=pz;
        home[i*3]=px; home[i*3+1]=py; home[i*3+2]=pz;
        dispersed[i*3] = (Math.random()-0.5)*28;
        dispersed[i*3+1] = (Math.random()-0.5)*28;
        dispersed[i*3+2] = (Math.random()-0.5)*28;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
      const mat = new THREE.PointsMaterial({
        color, size:0.032, transparent:true, opacity:0.45,
        blending:THREE.AdditiveBlending, depthWrite:false
      });
      const pts = new THREE.Points(geo, mat);
      pts.userData.home = home;
      pts.userData.dispersed = dispersed;
      pts.userData.baseOpacity = 0.45;
      molecule.add(pts);
      parts.push(pts);
      return pts;
    }

    // linear CO2 molecule: O = C = O
    makeCluster(480, 0.95, [0,0,0], 0xF5F1EA, 0.055, 0.85);
    makeCluster(320, 0.62, [-2.5,0,0], 0xFF6B35, 0.05, 0.85);
    makeCluster(320, 0.62, [2.5,0,0], 0xFF6B35, 0.05, 0.85);
    makeBond([-2.5,0,0], [-0.95,0,0], 140, 0x7C3AED);
    makeBond([0.95,0,0], [2.5,0,0], 140, 0x7C3AED);

    let mouseX = 0, mouseY = 0, autoRot = 0;
    window.addEventListener('mousemove', (e) => {
      mouseX = (e.clientX/window.innerWidth - 0.5);
      mouseY = (e.clientY/window.innerHeight - 0.5);
    }, { passive:true });

    function resize(){
      const w = container.clientWidth, h = Math.max(container.clientHeight,1);
      camera.aspect = w/h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener('resize', resize);

    function tick(){
      requestAnimationFrame(tick);
      autoRot += 0.0022;
      molecule.rotation.y = autoRot + mouseX*0.5;
      molecule.rotation.x += (mouseY*0.3 - molecule.rotation.x)*0.04;

      const progress = window.moleculeScrollProgress || 0;
      parts.forEach(pts => {
        const arr = pts.geometry.attributes.position.array;
        const home = pts.userData.home, disp = pts.userData.dispersed;
        for(let i=0;i<arr.length;i++){
          arr[i] = home[i] + (disp[i]-home[i])*progress;
        }
        pts.geometry.attributes.position.needsUpdate = true;
        pts.material.opacity = pts.userData.baseOpacity * (1 - progress*0.92);
      });

      renderer.render(scene, camera);
    }
    tick();
  })();