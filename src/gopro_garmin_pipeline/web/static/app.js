/* ═══════════════════════════════════════════════════════════════════
   CYCLING TELEMETRY HUD
   Left: HUD map + GPS trace   Right: Stacked stats   Top-right: Elevation
   ═══════════════════════════════════════════════════════════════════ */

let rideData = null;
let elevCanvas = null;
let routeCanvas = null;
let routeProjected = null;
let fontsReady = false;

let lMap = null;
let lMarker = null;

const TRACE_W = 220;
const TRACE_H = 140;
const TRACE_PAD = 14;
const EDGE_PAD = 18;

const C = {
    panelBg:   'rgba(0, 0, 0, 0.45)',
    panelEdge: 'rgba(255,255,255,0.06)',
    white:     'rgba(255,255,255,0.92)',
    dim:       'rgba(255,255,255,0.35)',
    label:     'rgba(255,255,255,0.45)',

    speed:     '#FFFFFF',
    hr:        '#FF6B6B',
    power:     '#FFD43B',
    cadence:   '#7ECFB3',

    traceGlow: 'rgba(255,255,255,0.10)',
    traceLine: 'rgba(255,255,255,0.45)',
    dot:       '#FFFFFF',

    elevBg:    'rgba(0, 0, 0, 0.45)',
    elevLine:  'rgba(255,255,255,0.40)',
    elevFill:  'rgba(255,255,255,0.06)',
    elevMark:  '#FF6B6B',
};

document.fonts.ready.then(() => { fontsReady = true; });

let videoSyncStarted = false;

function loadRideData(clipName) {
    const url = clipName ? `/api/ride-data?clip=${encodeURIComponent(clipName)}` : '/api/ride-data';
    return fetch(url)
        .then(r => r.json())
        .then(data => {
            rideData = data;
            const distTotalEl = document.getElementById('dist-total');
            if (distTotalEl) distTotalEl.textContent = data.total_distance_miles.toFixed(2);
            routeProjected = projectRoute(data.route, data.route_bounds);
            routeCanvas = buildRouteCanvas();
            elevCanvas = buildElevationCanvas(data.elevation_profile);
            if (!videoSyncStarted) {
                initHudMap(data);
                initVideoSync();
                videoSyncStarted = true;
            }
        });
}

// Initial load for the first clip.
loadRideData();


// ─── GPS Trace Projection ──────────────────────────────────────────

function projectRoute(route, bounds) {
    const drawW = TRACE_W - 2 * TRACE_PAD;
    const drawH = TRACE_H - 2 * TRACE_PAD;
    const latRange = bounds.max_lat - bounds.min_lat || 1e-6;
    const lonRange = bounds.max_lon - bounds.min_lon || 1e-6;
    const midLat = (bounds.min_lat + bounds.max_lat) / 2;
    const cosLat = Math.cos(midLat * Math.PI / 180);
    const routeAspect = (lonRange * cosLat) / latRange;
    const boxAspect = drawW / drawH;

    let xScale, yScale, xOff, yOff;
    if (routeAspect > boxAspect) {
        xScale = drawW; yScale = drawW / routeAspect;
        xOff = TRACE_PAD; yOff = TRACE_PAD + (drawH - yScale) / 2;
    } else {
        yScale = drawH; xScale = drawH * routeAspect;
        xOff = TRACE_PAD + (drawW - xScale) / 2; yOff = TRACE_PAD;
    }

    return route.map(([lat, lon]) => ({
        x: xOff + ((lon - bounds.min_lon) / lonRange) * xScale,
        y: yOff + ((bounds.max_lat - lat) / latRange) * yScale,
    }));
}

function projectPointOnTrace(lat, lon) {
    const bounds = rideData.route_bounds;
    const drawW = TRACE_W - 2 * TRACE_PAD;
    const drawH = TRACE_H - 2 * TRACE_PAD;
    const latRange = bounds.max_lat - bounds.min_lat || 1e-6;
    const lonRange = bounds.max_lon - bounds.min_lon || 1e-6;
    const midLat = (bounds.min_lat + bounds.max_lat) / 2;
    const cosLat = Math.cos(midLat * Math.PI / 180);
    const routeAspect = (lonRange * cosLat) / latRange;
    const boxAspect = drawW / drawH;

    let xScale, yScale, xOff, yOff;
    if (routeAspect > boxAspect) {
        xScale = drawW; yScale = drawW / routeAspect;
        xOff = TRACE_PAD; yOff = TRACE_PAD + (drawH - yScale) / 2;
    } else {
        yScale = drawH; xScale = drawH * routeAspect;
        xOff = TRACE_PAD + (drawW - xScale) / 2; yOff = TRACE_PAD;
    }

    return {
        x: xOff + ((lon - bounds.min_lon) / lonRange) * xScale,
        y: yOff + ((bounds.max_lat - lat) / latRange) * yScale,
    };
}

// ─── Offscreen GPS Trace ───────────────────────────────────────────

function buildRouteCanvas() {
    const c = document.createElement('canvas');
    c.width = TRACE_W; c.height = TRACE_H;
    const ctx = c.getContext('2d');

    ctx.beginPath();
    ctx.roundRect(0, 0, TRACE_W, TRACE_H, 10);
    ctx.clip();
    ctx.fillStyle = 'rgba(5, 5, 8, 0.65)';
    ctx.fillRect(0, 0, TRACE_W, TRACE_H);

    ctx.strokeStyle = C.panelEdge;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(0.5, 0.5, TRACE_W - 1, TRACE_H - 1, 10);
    ctx.stroke();

    if (routeProjected && routeProjected.length > 1) {
        ctx.strokeStyle = C.traceGlow;
        ctx.lineWidth = 5; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(routeProjected[0].x, routeProjected[0].y);
        for (let i = 1; i < routeProjected.length; i++)
            ctx.lineTo(routeProjected[i].x, routeProjected[i].y);
        ctx.stroke();

        ctx.strokeStyle = C.traceLine;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(routeProjected[0].x, routeProjected[0].y);
        for (let i = 1; i < routeProjected.length; i++)
            ctx.lineTo(routeProjected[i].x, routeProjected[i].y);
        ctx.stroke();
    }

    return c;
}

// ─── Leaflet HUD Map ───────────────────────────────────────────────

function initHudMap(data) {
    const route = data.route;
    if (!route || route.length < 2) return;

    const bounds = data.route_bounds;
    const centerLat = (bounds.min_lat + bounds.max_lat) / 2;
    const centerLon = (bounds.min_lon + bounds.max_lon) / 2;

    lMap = L.map('hud-map', {
        zoomControl: false, attributionControl: false,
        dragging: false, scrollWheelZoom: false,
        doubleClickZoom: false, touchZoom: false, keyboard: false,
    }).setView([centerLat, centerLon], 15);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
    }).addTo(lMap);

    L.polyline(route, { color: 'rgba(255,255,255,0.10)', weight: 5, lineCap: 'round', lineJoin: 'round' }).addTo(lMap);
    L.polyline(route, { color: 'rgba(255,255,255,0.5)', weight: 2, lineCap: 'round', lineJoin: 'round' }).addTo(lMap);

    const icon = L.divIcon({
        className: '',
        html: `<div style="width:14px;height:14px;background:#fff;border-radius:50%;box-shadow:0 0 12px 4px rgba(255,255,255,0.45),0 0 28px 8px rgba(255,255,255,0.15);border:2px solid rgba(0,0,0,0.2);"></div>`,
        iconSize: [14, 14], iconAnchor: [7, 7],
    });
    lMarker = L.marker([centerLat, centerLon], { icon }).addTo(lMap);
}

function updateHudMap(point) {
    if (!lMap || !lMarker || point.lat == null || point.lon == null) return;
    lMarker.setLatLng([point.lat, point.lon]);
    lMap.setView([point.lat, point.lon], 15, { animate: false });
}

// ─── Elevation Profile ─────────────────────────────────────────────

function buildElevationCanvas(profile) {
    if (!profile || profile.length < 2) return null;

    const W = 180, H = 100;
    const pad = { top: 22, right: 8, bottom: 16, left: 34 };
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');

    ctx.beginPath(); ctx.roundRect(0, 0, W, H, 8); ctx.clip();
    ctx.fillStyle = C.elevBg; ctx.fillRect(0, 0, W, H);

    const dists = profile.map(p => p[0]), alts = profile.map(p => p[1]);
    const minD = Math.min(...dists), maxD = Math.max(...dists);
    const minA = Math.min(...alts),  maxA = Math.max(...alts);
    const rangeD = maxD - minD || 1, rangeA = maxA - minA || 1;
    const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;

    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH);
    for (const [d, a] of profile)
        ctx.lineTo(pad.left + ((d - minD) / rangeD) * plotW, pad.top + plotH - ((a - minA) / rangeA) * plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.closePath();
    ctx.fillStyle = C.elevFill; ctx.fill();

    ctx.beginPath();
    for (let i = 0; i < profile.length; i++) {
        const x = pad.left + ((profile[i][0] - minD) / rangeD) * plotW;
        const y = pad.top + plotH - ((profile[i][1] - minA) / rangeA) * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = C.elevLine; ctx.lineWidth = 1.5; ctx.stroke();

    ctx.fillStyle = C.dim; ctx.font = '9px Rajdhani, sans-serif';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`${Math.round(maxA)}ft`, pad.left - 3, pad.top + 3);
    ctx.fillText(`${Math.round(minA)}ft`, pad.left - 3, pad.top + plotH - 3);

    ctx.strokeStyle = C.panelEdge; ctx.lineWidth = 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.beginPath(); ctx.roundRect(0.5, 0.5, W - 1, H - 1, 8); ctx.stroke();

    c._bounds = { minD, maxD, minA, maxA, rangeD, rangeA, plotW, plotH, pad };
    return c;
}

// ─── Binary Search ─────────────────────────────────────────────────

function findNearestPoint(videoTime) {
    const pts = rideData.points;
    if (!pts.length) return null;
    let lo = 0, hi = pts.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (pts[mid].t < videoTime) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && Math.abs(pts[lo - 1].t - videoTime) < Math.abs(pts[lo].t - videoTime))
        return pts[lo - 1];
    return pts[lo];
}

// ─── Video Sync ────────────────────────────────────────────────────

function initVideoSync() {
    const video  = document.getElementById('player');
    const canvas = document.getElementById('overlay');
    const ctx    = canvas.getContext('2d');

    function resize() {
        const rect = video.getBoundingClientRect();
        canvas.width  = rect.width  * devicePixelRatio;
        canvas.height = rect.height * devicePixelRatio;
        canvas.style.width  = rect.width  + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        update();
    }

    const ro = new ResizeObserver(resize);
    ro.observe(video);
    video.addEventListener('loadedmetadata', resize);

    function update() {
        if (!rideData || !fontsReady) return;
        const point = findNearestPoint(video.currentTime);
        if (!point) return;

        const rect = video.getBoundingClientRect();
        const w = rect.width, h = rect.height;

        ctx.clearRect(0, 0, w, h);
        drawRightStats(ctx, point, w, h);
        drawGpsTrace(ctx, point, w, h);
        drawDistance(ctx, point, w, h);
        drawElevation(ctx, point, w, h);

        updateHudMap(point);
        document.getElementById('dist-current').textContent =
            point.distance != null ? point.distance.toFixed(2) : '--';
    }

    function tick() {
        if (!video.paused && rideData) update();
        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    video.addEventListener('seeked', update);
    video.addEventListener('play', update);
}

// ═══════════════════════════════════════════════════════════════════
//  RIGHT SIDE STATS — stacked vertically, like the reference
// ═══════════════════════════════════════════════════════════════════

function drawRightStats(ctx, point, w, h) {
    // Build metrics list, skip gradient if zero/null
    const metrics = [
        { value: point.speed != null ? point.speed.toFixed(1) : '--',     unit: 'MPH',   color: C.speed },
        { value: point.power != null ? point.power.toString() : '--',     unit: 'W',     color: C.power },
        { value: point.hr    != null ? point.hr.toString()    : '--',     unit: 'BPM',   color: C.hr },
        { value: point.cadence != null ? point.cadence.toString() : '--', unit: 'RPM',   color: C.cadence },
    ];

    // Only show gradient if populated and non-zero
    if (point.gradient != null && Math.abs(point.gradient) > 0.1) {
        metrics.push({
            value: (point.gradient >= 0 ? '+' : '') + point.gradient.toFixed(1),
            unit: '%',
            color: '#B8A9E8',
        });
    }

    const lineH = 52;
    const totalH = metrics.length * lineH;
    const x = w - EDGE_PAD;
    const startY = (h - totalH) / 2;

    metrics.forEach((m, i) => {
        const y = startY + i * lineH + lineH / 2;

        // Large value — right-aligned
        ctx.fillStyle = m.color;
        ctx.font = '700 38px Orbitron, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(m.value, x - 50, y);

        // Unit — small, to the right of value, top-aligned
        ctx.fillStyle = C.label;
        ctx.font = '600 12px Rajdhani, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(m.unit, x - 44, y);
    });

    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
}

// ═══════════════════════════════════════════════════════════════════
//  GPS TRACE — bottom-left, below HUD map
// ═══════════════════════════════════════════════════════════════════

function drawGpsTrace(ctx, point, w, h) {
    if (!routeCanvas) return;

    const x = EDGE_PAD;
    const y = h - TRACE_H - 80;

    ctx.drawImage(routeCanvas, x, y);

    if (point.lat != null && point.lon != null) {
        const pos = projectPointOnTrace(point.lat, point.lon);
        const px = x + pos.x, py = y + pos.y;

        const grad = ctx.createRadialGradient(px, py, 0, px, py, 12);
        grad.addColorStop(0, 'rgba(255,255,255,0.5)');
        grad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = C.dot;
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fill();
    }
}

// ═══════════════════════════════════════════════════════════════════
//  DISTANCE — below GPS trace
// ═══════════════════════════════════════════════════════════════════

function drawDistance(ctx, point, w, h) {
    const current = point.distance != null ? point.distance.toFixed(2) : '--';
    const total = rideData.total_distance_miles.toFixed(2);

    const x = EDGE_PAD + TRACE_W / 2;
    const y = h - 68;

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = C.white;
    ctx.font = '700 16px Orbitron, monospace';
    ctx.fillText(current, x - 30, y);

    const cw = ctx.measureText(current).width;
    ctx.fillStyle = C.dim;
    ctx.font = '500 13px Rajdhani, sans-serif';
    ctx.fillText(`/ ${total} MI`, x + cw / 2 + 6, y);

    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
}

// ═══════════════════════════════════════════════════════════════════
//  ELEVATION — top right
// ═══════════════════════════════════════════════════════════════════

function drawElevation(ctx, point, w, h) {
    if (!elevCanvas) return;

    const eW = 180, x = w - eW - EDGE_PAD, y = EDGE_PAD;
    ctx.drawImage(elevCanvas, x, y);

    if (point.distance != null && elevCanvas._bounds) {
        const b = elevCanvas._bounds;
        const markerX = x + b.pad.left + ((point.distance - b.minD) / b.rangeD) * b.plotW;
        const clampedX = Math.max(x + b.pad.left, Math.min(markerX, x + b.pad.left + b.plotW));

        ctx.strokeStyle = C.elevMark; ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(clampedX, y + b.pad.top);
        ctx.lineTo(clampedX, y + b.pad.top + b.plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        if (point.altitude_ft != null) {
            const markerY = y + b.pad.top + b.plotH - ((point.altitude_ft - b.minA) / b.rangeA) * b.plotH;
            ctx.fillStyle = C.elevMark;
            ctx.beginPath(); ctx.arc(clampedX, markerY, 3, 0, Math.PI * 2); ctx.fill();

            ctx.fillStyle = C.white;
            ctx.font = '600 10px Rajdhani, sans-serif';
            ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText(`${Math.round(point.altitude_ft)}ft`, clampedX + 5, markerY - 2);
        }
    }
}
