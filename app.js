/* ----------------------------------------------------
   KISHMI AURA — INTERACTIVE CONTROLLER
   Vanilla JS Logic | Custom HUD & Scanning Sequence
   ---------------------------------------------------- */

// Set Product Catalog prices for the Ritual Bag
const PRODUCT_PRICES = {
  "Kishmi Clarifying Gel Cleanser": 680,
  "Kishmi Hydrating Foam Cleanser": 550,
  "Kishmi 10% Niacinamide Serum": 890,
  "Kishmi Lightweight Ceramide Gel": 750,
  "Kishmi Barrier Repair Gel": 820,
  "Kishmi Matte SPF 50 PA+++": 950,
  "Kishmi 2% Salicylic Acid BHA Exfoliant": 850,
  "Kishmi Cica & Zinc Spot Gel": 580
};

// Simulated cities for local environmental response
const INDIAN_CLIMATES = [
  { city: "Mumbai, IN", weather: "Humid • 32°C • UV Index: Extreme (10) • Hard Water" },
  { city: "New Delhi, IN", weather: "Dry Heat • 41°C • UV Index: Extreme (11) • High Dust" },
  { city: "Bengaluru, IN", weather: "Moderate • 28°C • UV Index: High (8) • Hard Water" },
  { city: "Chennai, IN", weather: "Intense Humidity • 35°C • UV Index: Extreme (10) • Sea Salinity" },
  { city: "Kolkata, IN", weather: "Damp Humid • 33°C • UV Index: Very High (9) • Air Congestion" }
];

// App States
let appState = {
  activeSubject: null,
  activeView: 'front', // 'front', 'left', 'right'
  isScanning: false,
  scanCompiled: false,
  cart: [],
  customImageBase64: null
};

// DOM Cache
const subjectGrid = document.getElementById('subjectGrid');
const scanMainImage = document.getElementById('scanMainImage');
const markersOverlay = document.getElementById('markersOverlay');
const laserLine = document.getElementById('laserLine');
const telemetryHud = document.getElementById('telemetryHud');
const startScanBtn = document.getElementById('startScanBtn');
const consoleLogs = document.getElementById('consoleLogs');
const scanCompleteBadge = document.getElementById('scanCompleteBadge');
const viewTabBtns = document.querySelectorAll('.tab-btn');
const activeCity = document.getElementById('activeCity');
const activeWeather = document.getElementById('activeWeather');

// Report DOMs
const analysisPlaceholder = document.getElementById('analysisPlaceholder');
const analysisResults = document.getElementById('analysisResults');
const reportStory = document.getElementById('reportStory');
const badgeSkinType = document.getElementById('badgeSkinType');
const badgeUndertone = document.getElementById('badgeUndertone');
const badgeFitzpatrick = document.getElementById('badgeFitzpatrick');
const reportNoticed = document.getElementById('reportNoticed');
const morningSteps = document.getElementById('morningSteps');
const nightSteps = document.getElementById('nightSteps');
const reportTips = document.getElementById('reportTips');
const reportWarning = document.getElementById('reportWarning');

// Cart DOMs
const cartOverlay = document.getElementById('cartOverlay');
const openCartBtn = document.getElementById('openCartBtn');
const closeCartBtn = document.getElementById('closeCartBtn');
const cartItemsList = document.getElementById('cartItemsList');
const cartCountBadge = document.getElementById('cartCountBadge');
const cartSubtotal = document.getElementById('cartSubtotal');
const cartTotal = document.getElementById('cartTotal');
const checkoutBtn = document.getElementById('checkoutBtn');
const addRitualToCartBtn = document.getElementById('addRitualToCartBtn');

// Custom Upload DOMs
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

// Toast Notification
const toastNotification = document.getElementById('toastNotification');
const toastMessage = document.getElementById('toastMessage');

// ----------------------------------------------------
// INITIALIZATION
// ----------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  // 1. Set a random Indian city climate on load
  const randomClimate = INDIAN_CLIMATES[Math.floor(Math.random() * INDIAN_CLIMATES.length)];
  activeCity.textContent = randomClimate.city;
  activeWeather.textContent = randomClimate.weather;

  // 2. Render clinical subjects list
  renderSubjects();

  // 3. Load default subject (Subject 1)
  loadSubject(SKIN_DATABASE[0]);

  // 4. Bind view tab switchers
  viewTabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const view = e.currentTarget.getAttribute('data-view');
      switchProfileView(view);
    });
  });

  // 5. Scan trigger
  startScanBtn.addEventListener('click', runAuraScan);

  // 6. Cart overlays
  openCartBtn.addEventListener('click', () => toggleCart(true));
  closeCartBtn.addEventListener('click', () => toggleCart(false));
  cartOverlay.addEventListener('click', (e) => {
    if (e.target === cartOverlay) toggleCart(false);
  });

  // 7. Add entire ritual to cart
  addRitualToCartBtn.addEventListener('click', addWholeRitualToCart);

  // 8. Checkout click
  checkoutBtn.addEventListener('click', handleCheckout);

  // 9. Custom Upload click
  uploadArea.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleCustomImageUpload);
  
  // HUD update timer
  setInterval(updateHudTelemetry, 1000);
});

// ----------------------------------------------------
// SUBJECTS LOADER
// ----------------------------------------------------
function renderSubjects() {
  subjectGrid.innerHTML = '';
  SKIN_DATABASE.forEach(subject => {
    const card = document.createElement('div');
    card.className = `subject-card ${appState.activeSubject && appState.activeSubject.id === subject.id ? 'active' : ''}`;
    card.setAttribute('data-id', subject.id);
    
    // Custom inline background styling for avatar representation
    card.innerHTML = `
      <div class="subject-avatar" style="background-image: url('${subject.frontImage}')"></div>
      <div class="subject-details">
        <h4>Subject ${subject.id} (${subject.gender})</h4>
        <span class="subject-meta">${subject.skinType.split(' ')[0]} • ${subject.fitzpatrick}</span>
      </div>
    `;
    
    card.addEventListener('click', () => {
      if (appState.isScanning) return;
      const loadedSub = SKIN_DATABASE.find(s => s.id === subject.id);
      loadSubject(loadedSub);
    });
    
    subjectGrid.appendChild(card);
  });
}

function loadSubject(subject) {
  appState.activeSubject = subject;
  appState.activeView = 'front';
  appState.scanCompiled = false;
  appState.customImageBase64 = null;
  
  // Highlight active card
  document.querySelectorAll('.subject-card').forEach(card => {
    if (parseInt(card.getAttribute('data-id')) === subject.id) {
      card.classList.add('active');
    } else {
      card.classList.remove('active');
    }
  });

  // Update tabs visual state
  viewTabBtns.forEach(btn => {
    if (btn.getAttribute('data-view') === 'front') {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Load front photo
  scanMainImage.src = subject.frontImage;
  
  // Reset HUD
  document.getElementById('hudFitzpatrick').textContent = "CALIBRATING";
  document.getElementById('hudUnderEye').textContent = "READY";
  
  // Hide compiled report & markers
  analysisResults.classList.add('hidden');
  analysisPlaceholder.classList.remove('hidden');
  markersOverlay.innerHTML = '';
  scanCompleteBadge.classList.remove('show');
  
  // Write to console log
  logEntry(`[SYSTEM] Loaded Subject ${subject.id} profile scan. Ready for analysis.`, 'system');
}

// ----------------------------------------------------
// PROFILE SWITCH VIEWS
// ----------------------------------------------------
function switchProfileView(view) {
  if (appState.isScanning) return;
  appState.activeView = view;
  
  // Toggle active tab buttons
  viewTabBtns.forEach(btn => {
    if (btn.getAttribute('data-view') === view) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  const subject = appState.activeSubject;
  
  // Set correct image source
  if (appState.customImageBase64) {
    scanMainImage.src = appState.customImageBase64;
  } else if (subject) {
    if (view === 'front') scanMainImage.src = subject.frontImage;
    else if (view === 'left') scanMainImage.src = subject.leftImage;
    else if (view === 'right') scanMainImage.src = subject.rightImage;
  }


  // Redraw concern markers if scan is compiled
  if (appState.scanCompiled) {
    drawConcernMarkers();
  }
}

// ----------------------------------------------------
// SCAN SEQUENCE
// ----------------------------------------------------
function runAuraScan() {
  if (appState.isScanning) return;
  
  appState.isScanning = true;
  appState.scanCompiled = false;
  
  // UI Setup during scan
  laserLine.style.display = 'block';
  scanCompleteBadge.classList.remove('show');
  markersOverlay.innerHTML = '';
  analysisResults.classList.add('hidden');
  analysisPlaceholder.classList.remove('hidden');
  
  // Disable buttons
  startScanBtn.disabled = true;
  startScanBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> SCANNING SKIN LAYERS...`;
  
  consoleLogs.innerHTML = '';
  logEntry(`[SYSTEM] Initializing AURA skin boundary calibration grid...`, 'system');
  
  // Logging Sequence timers
  setTimeout(() => {
    logEntry(`[SPECTRAL] Scanning lipid barrier and sebum activity levels...`, 'process');
    document.getElementById('hudFitzpatrick').textContent = "CALCULATING";
  }, 600);

  setTimeout(() => {
    const sub = appState.activeSubject;
    logEntry(`[CLINICAL] Calibrating Fitzpatrick Scale: Matched ${sub.fitzpatrick} with ${sub.undertone} undertone.`, 'process');
    document.getElementById('hudFitzpatrick').textContent = sub.fitzpatrick;
  }, 1400);

  setTimeout(() => {
    logEntry(`[ANALYSIS] Deep mapping pore boundaries, active breakouts, and hydration balances...`, 'process');
    document.getElementById('hudUnderEye').textContent = "COMPILING";
  }, 2200);

  setTimeout(() => {
    // Scan compilation complete
    appState.isScanning = false;
    appState.scanCompiled = true;
    
    laserLine.style.display = 'none';
    scanCompleteBadge.classList.add('show');
    document.getElementById('hudUnderEye').textContent = "COMPILED";
    
    // Enable buttons
    startScanBtn.disabled = false;
    startScanBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> START AURA AI SCAN`;
    
    logEntry(`[SUCCESS] AURA AI Skin Analysis complete! Skin Story compiled.`, 'success');
    
    // Display reports & draw pins
    renderAuraReport();
    drawConcernMarkers();
  }, 3000);
}

// Telemetry Hud details
function updateHudTelemetry() {
  if (!appState.isScanning) return;
  const resolutionItems = ["1024x1024", "1920x1080", "2048x2048"];
  const resVal = resolutionItems[Math.floor(Math.random() * resolutionItems.length)];
  document.querySelector('.telemetry-hud .hud-item:nth-child(2) .hud-val').textContent = resVal;
  
  const fpsVal = (58.5 + Math.random() * 2.5).toFixed(1);
  document.querySelector('.telemetry-hud .hud-item:nth-child(1) .hud-val').textContent = fpsVal;
}

// ----------------------------------------------------
// RENDER ANALYSIS REPORT
// ----------------------------------------------------
function renderAuraReport() {
  const subject = appState.activeSubject;
  if (!subject) return;

  // Reveal Report Panel
  analysisPlaceholder.classList.add('hidden');
  analysisResults.classList.remove('hidden');

  // 1. Skin Story
  reportStory.innerHTML = subject.story;

  // 2. Badges & What We Noticed
  badgeSkinType.textContent = subject.skinType;
  badgeUndertone.textContent = subject.undertone;
  badgeFitzpatrick.textContent = subject.fitzpatrick;

  reportNoticed.innerHTML = '';
  subject.noticed.forEach(item => {
    const li = document.createElement('li');
    // Format bold strings if present
    li.innerHTML = formatMarkdownBold(item);
    reportNoticed.appendChild(li);
  });

  // 3. Skincare Ritual morning
  morningSteps.innerHTML = '';
  subject.ritual.morning.forEach(step => {
    morningSteps.appendChild(createRoutineStepNode(step));
  });

  // Night steps
  nightSteps.innerHTML = '';
  subject.ritual.night.forEach(step => {
    nightSteps.appendChild(createRoutineStepNode(step));
  });

  // 4. Climate Skin Tips
  reportTips.innerHTML = '';
  subject.tips.forEach(tip => {
    const li = document.createElement('li');
    li.textContent = tip;
    reportTips.appendChild(li);
  });

  // 5. Clinical warnings
  reportWarning.textContent = subject.dermWarning;
}

function createRoutineStepNode(step) {
  const price = PRODUCT_PRICES[step.product] || 750;
  const li = document.createElement('li');
  li.className = 'routine-step-item';
  li.innerHTML = `
    <div class="step-left">
      <span class="product-name">${step.product}</span>
      <span class="product-benefit">${step.benefit} • ₹${price}</span>
    </div>
    <button class="add-step-cart-btn" title="Add to Bag" onclick="addToCart('${step.product}', ${price})">
      <i class="fa-solid fa-cart-plus"></i>
    </button>
  `;
  return li;
}

// ----------------------------------------------------
// CONCERN PINPOINT MARKERS OVERLAY
// ----------------------------------------------------
function drawConcernMarkers() {
  markersOverlay.innerHTML = '';
  const subject = appState.activeSubject;
  if (!subject || !subject.markers) return;

  const activeViewMarkers = subject.markers[appState.activeView];
  if (!activeViewMarkers) return;

  activeViewMarkers.forEach(marker => {
    const pin = document.createElement('div');
    
    // Assign styling categories based on keywords
    let category = '';
    const nameLower = marker.concern.toLowerCase();
    if (nameLower.includes('breakout') || nameLower.includes('papule') || nameLower.includes('acne')) {
      category = 'breakout';
    } else if (nameLower.includes('pore') || nameLower.includes('congestion') || nameLower.includes('shine') || nameLower.includes('comedone')) {
      category = 'oily';
    }

    pin.className = `concern-marker ${category}`;
    pin.style.left = `${marker.x}%`;
    pin.style.top = `${marker.y}%`;
    pin.setAttribute('data-tooltip', `${marker.concern}: ${marker.desc}`);
    
    markersOverlay.appendChild(pin);
  });
}

// Helper to log telemetry console logs
function logEntry(text, type = 'system') {
  const div = document.createElement('div');
  div.className = `log-entry ${type}`;
  div.textContent = text;
  consoleLogs.appendChild(div);
  consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

// Markdown bold replacement helper
function formatMarkdownBold(text) {
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// ----------------------------------------------------
// SHOPPING CART DRAWER ACTIONS
// ----------------------------------------------------
function toggleCart(isOpen) {
  if (isOpen) {
    cartOverlay.classList.add('open');
    renderCartItems();
  } else {
    cartOverlay.classList.remove('open');
  }
}

window.addToCart = function(productName, price) {
  // Check if already in cart
  const exists = appState.cart.find(item => item.name === productName);
  if (exists) {
    showToast(`${productName} is already in your bag!`);
    return;
  }

  appState.cart.push({ name: productName, price: price });
  updateCartBadge();
  showToast(`Added ${productName} to your ritual bag!`);
};

function addWholeRitualToCart() {
  const subject = appState.activeSubject;
  if (!subject) return;

  let addedCount = 0;
  
  // Grab all morning products
  subject.ritual.morning.forEach(step => {
    const price = PRODUCT_PRICES[step.product] || 750;
    const exists = appState.cart.find(item => item.name === step.product);
    if (!exists) {
      appState.cart.push({ name: step.product, price: price });
      addedCount++;
    }
  });

  // Grab all night products
  subject.ritual.night.forEach(step => {
    const price = PRODUCT_PRICES[step.product] || 750;
    const exists = appState.cart.find(item => item.name === step.product);
    if (!exists) {
      appState.cart.push({ name: step.product, price: price });
      addedCount++;
    }
  });

  updateCartBadge();
  if (addedCount > 0) {
    showToast(`Added ${addedCount} ritual products to your bag!`);
    toggleCart(true);
  } else {
    showToast(`All ritual products are already in your bag!`);
  }
}

function renderCartItems() {
  cartItemsList.innerHTML = '';
  if (appState.cart.length === 0) {
    cartItemsList.innerHTML = `
      <div class="empty-cart-state">
        <i class="fa-solid fa-seedling"></i>
        <p>Your skincare routine bag is currently empty. Scan a subject and add products to start your ritual journey.</p>
      </div>
    `;
    updateCartTotals(0);
    return;
  }

  let subtotal = 0;
  appState.cart.forEach((item, index) => {
    subtotal += item.price;
    const div = document.createElement('div');
    div.className = 'cart-item';
    div.innerHTML = `
      <div class="cart-item-left">
        <span class="cart-item-title">${item.name}</span>
        <span class="cart-item-price">₹${item.price}.00</span>
      </div>
      <button class="remove-cart-item-btn" onclick="removeCartItem(${index})" title="Remove"><i class="fa-solid fa-trash-can"></i></button>
    `;
    cartItemsList.appendChild(div);
  });

  updateCartTotals(subtotal);
}

window.removeCartItem = function(index) {
  const item = appState.cart[index];
  appState.cart.splice(index, 1);
  updateCartBadge();
  renderCartItems();
  showToast(`Removed ${item.name} from bag.`);
};

function updateCartBadge() {
  cartCountBadge.textContent = appState.cart.length;
}

function updateCartTotals(subtotal) {
  cartSubtotal.textContent = `₹${subtotal}.00`;
  // Set a standard ₹150 brand discount if items exist, otherwise 0
  const discount = subtotal > 0 ? 150 : 0;
  document.querySelector('.discount-price').textContent = `-₹${discount}.00`;
  
  const finalTotal = Math.max(0, subtotal - discount);
  cartTotal.textContent = `₹${finalTotal}.00`;
}

function handleCheckout() {
  if (appState.cart.length === 0) {
    showToast("Please add products to your ritual bag before checking out.");
    return;
  }

  alert("🌟 Thank you for ordering your custom KISHMI Skincare Ritual! Your skin is on its way to clinical, glowing health. 🌿");
  appState.cart = [];
  updateCartBadge();
  toggleCart(false);
}

// ----------------------------------------------------
// DYNAMIC TOAST NOTIFICATIONS
// ----------------------------------------------------
function showToast(message) {
  toastMessage.textContent = message;
  toastNotification.classList.add('show');
  
  setTimeout(() => {
    toastNotification.classList.remove('show');
  }, 2500);
}

// ----------------------------------------------------
// CUSTOM IMAGE FILE UPLOADER
// ----------------------------------------------------
const API_ENDPOINT = "/analyze"; // Relative path for monolithic hosting

function handleCustomImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async function(event) {
    const base64Data = event.target.result;
    appState.customImageBase64 = base64Data;
    
    logEntry(`[SYSTEM] Uploading image to AI…`, 'system');
    showToast("Analyzing image with Custom YOLO AI...");
    
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Data })
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        const customSubject = {
            id: 999,
            gender: 'custom',
            frontImage: base64Data,
            leftImage: base64Data,
            rightImage: base64Data,
            skinType: "Custom Scanned Profile",
            undertone: "Auto-detected",
            fitzpatrick: "Auto-detected",
            story: data.story,
            noticed: data.noticed,
            ritual: data.ritual,
            tips: data.tips,
            dermWarning: data.dermWarning,
            markers: {
                front: data.markers,
                left: data.markers,
                right: data.markers
            }
        };
        
        loadSubject(customSubject);
        logEntry('[SUCCESS] AI analysis complete. Click START AURA AI SCAN to view results.', 'success');
        showToast("Scan ready! Click START AURA AI SCAN.");
        
    } catch (err) {
        console.error(err);
        logEntry(`[ERROR] ${err.message}`, 'error');
        showToast('AI analysis failed. Is the server running?');
    }
  };
  
  reader.readAsDataURL(file);
}

// ----------------------------------------------------
// LIVE WEBCAM AR SCANNER
// ----------------------------------------------------
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const capturePhotoBtn = document.getElementById('capturePhotoBtn');
const cameraControls = document.getElementById('cameraControls');
const webcamVideo = document.getElementById('webcamVideo');
const captureCanvas = document.getElementById('captureCanvas');

let webcamStream = null;

if (startCameraBtn) {
  startCameraBtn.addEventListener('click', async () => {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      webcamVideo.srcObject = webcamStream;
      
      // UI toggles
      scanMainImage.style.display = 'none';
      webcamVideo.style.display = 'block';
      cameraControls.style.display = 'flex';
      startCameraBtn.style.pointerEvents = 'none';
      startCameraBtn.style.opacity = '0.5';
      
      logEntry('[SYSTEM] Camera initialized. Ready to capture.', 'system');
      showToast("Camera Active. Take a photo!");
      
      // Clear old markers and results
      markersOverlay.innerHTML = '';
      document.querySelector('.analysis-placeholder').classList.add('hidden');
      document.getElementById('analysisResults').classList.remove('hidden');
      
      // Setup complete. Wait for user to click capture.
    } catch (err) {
      console.error(err);
      logEntry('[ERROR] Could not access webcam. Check permissions.', 'error');
      showToast("Camera access denied!");
    }
  });
}

if (stopCameraBtn) {
  stopCameraBtn.addEventListener('click', () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
    }
    
    scanMainImage.style.display = 'block';
    webcamVideo.style.display = 'none';
    cameraControls.style.display = 'none';
    startCameraBtn.style.pointerEvents = 'auto';
    startCameraBtn.style.opacity = '1';
    
    markersOverlay.innerHTML = '';
    logEntry('[SYSTEM] Camera closed.', 'system');
  });
}

if (capturePhotoBtn) {
    capturePhotoBtn.addEventListener('click', async () => {
        if (!webcamVideo.videoWidth) return;
        
        // Capture frame
        captureCanvas.width = webcamVideo.videoWidth;
        captureCanvas.height = webcamVideo.videoHeight;
        const ctx = captureCanvas.getContext('2d');
        
        ctx.translate(captureCanvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
        
        const base64Data = captureCanvas.toDataURL('image/jpeg', 0.9);
        
        // Display on main image
        scanMainImage.src = base64Data;
        scanMainImage.style.transform = 'scaleX(-1)';
        scanMainImage.style.display = 'block';
        
        // Stop Camera
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
        }
        webcamVideo.style.display = 'none';
        cameraControls.style.display = 'none';
        startCameraBtn.style.pointerEvents = 'auto';
        startCameraBtn.style.opacity = '1';
        
        logEntry('[SYSTEM] Photo captured. Analyzing...', 'system');
        showToast("Analyzing photo...");
        
        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data })
            });
            
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            
            appState.scanCompiled = true;
            const customSubject = {
                id: 999,
                skinType: "Uploaded Photo",
                undertone: "Auto Detected",
                fitzpatrick: "Auto Detected",
                frontImage: scanMainImage.src,
                story: data.story,
                noticed: data.noticed,
                ritual: data.ritual,
                tips: data.tips,
                dermWarning: data.dermWarning,
                markers: { front: data.markers, left: data.markers, right: data.markers }
            };
            
            appState.currentSubject = customSubject;
            loadSubject(customSubject);
            
            logEntry('[SUCCESS] Analysis complete.', 'success');
            showToast("Analysis complete!");
            
        } catch (err) {
            console.error("Capture Scan error:", err);
            logEntry(`[ERROR] ${err.message}`, 'error');
            showToast("Analysis failed!");
        }
    });
}

function drawLiveMarkers(markers) {
    markersOverlay.innerHTML = '';
    markers.forEach(marker => {
        const dot = document.createElement('div');
        dot.className = 'concern-marker';
        
        if (marker.concern === 'Acne') dot.classList.add('breakout');
        else if (marker.concern === 'Oily Skin') dot.classList.add('oily');
        
        // Invert X coordinate because video is mirrored via CSS
        let x = 100 - marker.x;
        
        dot.style.left = `${x}%`;
        dot.style.top = `${marker.y}%`;
        dot.setAttribute('data-tooltip', `${marker.concern} (${marker.desc})`);
        
        markersOverlay.appendChild(dot);
    });
}
