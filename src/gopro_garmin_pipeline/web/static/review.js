/* ═══════════════════════════════════════════════════════════════════
   Ride Review — moment list, clip switcher, ribbon + ride overview.

   Active when body[data-mode="ride"]. Coordinates with the existing
   single-clip telemetry app.js: this file owns the sidebar, ribbons,
   and clip switching; app.js owns the video + telemetry HUD.
   ═══════════════════════════════════════════════════════════════════ */

const REVIEW = {
    clips: [],            // [{filename, duration_secs, ride_start_secs}]
    moments: [],          // [MomentProposal, ...]
    rideDurationSecs: 0,
    currentClip: null,
    currentClipMeta: null,
    activeMomentId: null,
    filter: 'all',
    drag: null,           // { id, kind: 'in'|'out'|'anchor', startX, ... }
};

// ─── Bootstrap ─────────────────────────────────────────────────────

(async function init() {
    if (document.body.dataset.mode !== 'ride') return;
    await Promise.all([loadClips(), loadMoments(), loadRideInfo()]);
    wireControls();
    wireRibbonInteractions();
    wirePlayheadSync();
    wireKeyboardShortcuts();
    renderAll();
})();

async function loadClips() {
    const r = await fetch('/api/clips');
    REVIEW.clips = await r.json();
    REVIEW.currentClip = REVIEW.clips[0]?.filename || null;
    REVIEW.currentClipMeta = REVIEW.clips[0] || null;
    const sel = document.getElementById('clip-selector');
    sel.innerHTML = REVIEW.clips
        .map(c => `<option value="${c.filename}">${c.filename} (${formatTime(c.duration_secs)})</option>`)
        .join('');
}

async function loadMoments() {
    const r = await fetch('/api/moments');
    REVIEW.moments = await r.json();
}

async function loadRideInfo() {
    const r = await fetch('/api/ride-info');
    const data = await r.json();
    REVIEW.rideDurationSecs = data.ride_duration_secs || 0;
}

function renderAll() {
    renderSidebar();
    renderClipRibbon();
    renderRideStrip();
}

// ─── Wiring ────────────────────────────────────────────────────────

function wireControls() {
    document.getElementById('clip-selector').addEventListener('change', (e) => {
        switchClip(e.target.value);
    });

    document.querySelectorAll('input[name="filter"]').forEach(el => {
        el.addEventListener('change', (e) => {
            REVIEW.filter = e.target.value;
            renderSidebar();
        });
    });

    document.getElementById('save-selections').addEventListener('click', async () => {
        const r = await fetch('/api/save-selections', { method: 'POST' });
        const data = await r.json();
        toast(`Saved ${data.saved} approved moments`);
    });

    document.getElementById('add-at-playhead').addEventListener('click', addAtPlayhead);
}

function wirePlayheadSync() {
    const player = document.getElementById('player');
    const update = () => {
        const t = player.currentTime || 0;
        const dur = REVIEW.currentClipMeta?.duration_secs || 1;
        const clipPct = Math.max(0, Math.min(100, (t / dur) * 100));
        const clipHead = document.getElementById('clip-ribbon-playhead');
        if (clipHead) clipHead.style.left = clipPct + '%';

        const ride = REVIEW.rideDurationSecs;
        if (ride > 0 && REVIEW.currentClipMeta) {
            const rideT = REVIEW.currentClipMeta.ride_start_secs + t;
            const ridePct = Math.max(0, Math.min(100, (rideT / ride) * 100));
            const rideHead = document.getElementById('ride-strip-playhead');
            if (rideHead) rideHead.style.left = ridePct + '%';
        }
    };
    player.addEventListener('timeupdate', update);
    player.addEventListener('seeked', update);
    player.addEventListener('loadedmetadata', update);
}

function wireKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ignore when typing in an input/textarea
        if (e.target.matches('input, textarea, select')) return;

        const player = document.getElementById('player');
        const t = player.currentTime || 0;
        const m = REVIEW.moments.find(x => x.stable_id === REVIEW.activeMomentId);

        switch (e.key) {
            case 'a': if (m) toggleStatus(m, 'approved'); break;
            case 'r': if (m) toggleStatus(m, 'rejected'); break;
            case '[': if (m) setInOut(m, t, null); break;
            case ']': if (m) setInOut(m, null, t); break;
            case 'c': if (m) setAnchor(m, t); break;
            case 'n': addAtPlayhead(); break;
            case 'j': stepActiveMoment(-1); break;
            case 'k': stepActiveMoment(+1); break;
            default: return;
        }
        e.preventDefault();
    });
}

// ─── Per-clip ribbon ───────────────────────────────────────────────

function wireRibbonInteractions() {
    const track = document.getElementById('clip-ribbon-track');
    if (!track) return;

    // Click empty track area → seek video (skipping clicks on ticks/handles).
    track.addEventListener('click', (e) => {
        if (e.target.closest('.ribbon-tick, .ribbon-handle, .ribbon-band')) return;
        const dur = REVIEW.currentClipMeta?.duration_secs || 0;
        if (!dur) return;
        const rect = track.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        document.getElementById('player').currentTime = Math.max(0, Math.min(dur, pct * dur));
    });
}

function renderClipRibbon() {
    const track = document.getElementById('clip-ribbon-track');
    if (!track || !REVIEW.currentClipMeta) return;

    // Wipe everything except the playhead.
    Array.from(track.querySelectorAll('.ribbon-tick, .ribbon-band, .ribbon-handle'))
        .forEach(el => el.remove());

    const dur = REVIEW.currentClipMeta.duration_secs || 1;
    const inClip = REVIEW.moments.filter(m => m.clip_name === REVIEW.currentClip);
    const active = REVIEW.moments.find(m => m.stable_id === REVIEW.activeMomentId);

    inClip.forEach(m => {
        const anchor = effectiveAnchor(m);
        const tick = document.createElement('div');
        tick.className = `ribbon-tick status-${m.status || 'pending'}`;
        if (m.stable_id === REVIEW.activeMomentId) tick.classList.add('is-active');
        tick.style.left = (anchor / dur * 100) + '%';
        tick.dataset.id = m.stable_id;
        tick.title = `${formatTime(m.ride_time_secs)} · ${m.notes || m.sources.join('+')}`;
        tick.addEventListener('click', (e) => {
            e.stopPropagation();
            setActive(m.stable_id);
            seekTo(m.clip_name, anchor);
        });
        track.appendChild(tick);
    });

    // Active moment band + handles (only if active is in current clip)
    if (active && active.clip_name === REVIEW.currentClip) {
        const [vIn, vOut] = effectiveSpan(active);
        const anchor = effectiveAnchor(active);
        const band = document.createElement('div');
        band.className = 'ribbon-band';
        band.style.left = (vIn / dur * 100) + '%';
        band.style.width = ((vOut - vIn) / dur * 100) + '%';
        track.appendChild(band);

        track.appendChild(makeHandle('in', vIn, dur, active.stable_id));
        track.appendChild(makeHandle('anchor', anchor, dur, active.stable_id));
        track.appendChild(makeHandle('out', vOut, dur, active.stable_id));
    }
}

function makeHandle(kind, time, dur, momentId) {
    const h = document.createElement('div');
    h.className = `ribbon-handle handle-${kind}`;
    h.style.left = (time / dur * 100) + '%';
    h.dataset.kind = kind;
    h.dataset.id = momentId;
    const label = document.createElement('span');
    label.className = 'handle-label';
    label.textContent = kind === 'anchor' ? 'cut' : kind;
    h.appendChild(label);
    h.addEventListener('mousedown', (e) => beginDrag(e, kind, momentId));
    return h;
}

function beginDrag(e, kind, momentId) {
    e.preventDefault();
    e.stopPropagation();
    const track = document.getElementById('clip-ribbon-track');
    const rect = track.getBoundingClientRect();
    REVIEW.drag = { kind, momentId, rect, startedAt: Date.now() };

    const onMove = (ev) => updateDrag(ev);
    const onUp = (ev) => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        finishDrag(ev);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
}

function updateDrag(e) {
    const d = REVIEW.drag;
    if (!d) return;
    const m = REVIEW.moments.find(x => x.stable_id === d.momentId);
    if (!m) return;
    const dur = REVIEW.currentClipMeta?.duration_secs || 1;
    let t = ((e.clientX - d.rect.left) / d.rect.width) * dur;
    t = Math.max(0, Math.min(dur, t));

    // Constrain so in <= anchor <= out
    const [curIn, curOut] = effectiveSpan(m);
    const curAnchor = effectiveAnchor(m);
    if (d.kind === 'in')      t = Math.min(t, curAnchor);
    if (d.kind === 'out')     t = Math.max(t, curAnchor);
    if (d.kind === 'anchor')  t = Math.max(curIn, Math.min(curOut, t));

    // Live preview by mutating the moment locally; persist on mouseup.
    if (d.kind === 'anchor') {
        m.user_anchor_override = t;
    } else {
        const [vIn, vOut] = effectiveSpan(m);
        m.user_in_out = (d.kind === 'in') ? [t, vOut] : [vIn, t];
    }
    renderClipRibbon();
    // Live-seek the player to the dragged handle so the user sees what they're picking
    document.getElementById('player').currentTime = t;
}

async function finishDrag() {
    const d = REVIEW.drag;
    REVIEW.drag = null;
    if (!d) return;
    const m = REVIEW.moments.find(x => x.stable_id === d.momentId);
    if (!m) return;
    const body = (d.kind === 'anchor')
        ? { user_anchor_override: m.user_anchor_override }
        : { user_in_out: m.user_in_out };
    await postUpdate(m.stable_id, body);
    renderSidebar();  // sidebar shows updated time/notes
}

// ─── Ride-wide overview strip ──────────────────────────────────────

function renderRideStrip() {
    const track = document.getElementById('ride-strip-track');
    if (!track || !REVIEW.rideDurationSecs) return;

    Array.from(track.querySelectorAll('.ride-strip-clip, .ride-strip-tick'))
        .forEach(el => el.remove());

    const ride = REVIEW.rideDurationSecs;

    REVIEW.clips.forEach(c => {
        const block = document.createElement('div');
        block.className = 'ride-strip-clip';
        if (c.filename === REVIEW.currentClip) block.classList.add('is-current');
        block.style.left = (c.ride_start_secs / ride * 100) + '%';
        block.style.width = (c.duration_secs / ride * 100) + '%';
        block.title = `${c.filename} (${formatTime(c.duration_secs)})`;
        track.appendChild(block);
    });

    REVIEW.moments.forEach(m => {
        const tick = document.createElement('div');
        tick.className = `ride-strip-tick status-${m.status || 'pending'}`;
        if (m.stable_id === REVIEW.activeMomentId) tick.classList.add('is-active');
        tick.style.left = (m.ride_time_secs / ride * 100) + '%';
        track.appendChild(tick);
    });

    track.onclick = (e) => {
        if (e.target.closest('.ride-strip-tick')) return;
        const rect = track.getBoundingClientRect();
        const t = ((e.clientX - rect.left) / rect.width) * ride;
        const clip = clipContainingRideTime(t);
        if (clip) {
            const localT = t - clip.ride_start_secs;
            seekTo(clip.filename, localT);
        }
    };
}

function clipContainingRideTime(rideSecs) {
    return REVIEW.clips.find(c =>
        rideSecs >= c.ride_start_secs &&
        rideSecs <= c.ride_start_secs + c.duration_secs
    ) || REVIEW.clips[0];
}

// ─── Sidebar ────────────────────────────────────────────────────────

function renderSidebar() {
    const list = document.getElementById('moments-list');
    const filtered = REVIEW.moments
        .filter(m => REVIEW.filter === 'all'
            || (REVIEW.filter === 'approved' && m.status === 'approved')
            || (REVIEW.filter === 'pending' && m.status === 'pending'))
        .slice()
        .sort((a, b) => a.ride_time_secs - b.ride_time_secs);

    list.innerHTML = filtered.map(renderMomentRow).join('');
    list.querySelectorAll('.moment').forEach(li => {
        li.addEventListener('click', (e) => {
            if (e.target.closest('button')) return;
            const id = li.dataset.id;
            const m = REVIEW.moments.find(x => x.stable_id === id);
            if (!m) return;
            setActive(id);
            seekTo(m.clip_name, effectiveAnchor(m));
        });
        li.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                handleAction(li.dataset.id, btn.dataset.action, btn.dataset.value);
            });
        });
    });
    renderSummary();
    renderRideStrip();   // tick states track sidebar state
    renderClipRibbon();  // and the ribbon
}

function renderMomentRow(m) {
    const id = m.stable_id;
    const time = formatTime(m.ride_time_secs);
    const sources = (m.sources || []).join('+') || '?';
    const score = (m.score ?? 0).toFixed(1);
    const status = m.status || 'pending';
    const notes = escapeHtml(m.notes || '');
    const isActive = REVIEW.activeMomentId === id;

    const rubricParts = [];
    const r = m.rubric || {};
    if (r.light != null)       rubricParts.push(`L${r.light}`);
    if (r.composition != null) rubricParts.push(`C${r.composition}`);
    if (r.motion != null)      rubricParts.push(`M${r.motion}`);
    if (r.scenery != null)     rubricParts.push(`S${r.scenery}`);
    if (r.subject != null)     rubricParts.push(`👁${r.subject}`);

    const ratingButtons = Array.from({length: 10}, (_, i) => {
        const v = i + 1;
        const cls = (m.rating === v) ? 'is-active' : '';
        return `<button class="${cls}" data-action="rating" data-value="${v}">${v}</button>`;
    }).join('');

    const userEdited = (m.user_anchor_override != null || m.user_in_out != null) ? '✎ ' : '';

    return `
        <li class="moment ${isActive ? 'is-active' : ''}" data-id="${id}">
            <div class="moment-row">
                <span class="moment-time">${userEdited}${time}</span>
                <span class="moment-status ${status}">${status}</span>
                <span class="moment-score">${score}</span>
            </div>
            <div class="moment-row">
                <span class="moment-source">${sources}</span>
                ${rubricParts.length ? `<span class="moment-rubric">${rubricParts.join(' · ')}</span>` : ''}
            </div>
            <div class="moment-notes">${notes}</div>
            <div class="moment-actions">
                <button class="btn approve ${status === 'approved' ? 'is-active' : ''}"
                        data-action="status" data-value="approved">Approve</button>
                <button class="btn reject ${status === 'rejected' ? 'is-active' : ''}"
                        data-action="status" data-value="rejected">Reject</button>
            </div>
            <div class="rating-strip">${ratingButtons}</div>
        </li>
    `;
}

function renderSummary() {
    const total = REVIEW.moments.length;
    const approved = REVIEW.moments.filter(m => m.status === 'approved').length;
    const rejected = REVIEW.moments.filter(m => m.status === 'rejected').length;
    const pending = total - approved - rejected;
    const dur = REVIEW.moments
        .filter(m => m.status === 'approved')
        .reduce((s, m) => s + (m.final_trim_secs ?? 3), 0);
    document.getElementById('sidebar-summary').textContent =
        `${approved} approved · ${pending} pending · ${rejected} rejected · ${dur.toFixed(0)}s of cut`;
}

// ─── Actions ───────────────────────────────────────────────────────

async function handleAction(id, action, value) {
    const m = REVIEW.moments.find(x => x.stable_id === id);
    if (!m) return;
    if (action === 'status') {
        await toggleStatus(m, value);
    } else if (action === 'rating') {
        const v = parseInt(value, 10);
        await postUpdate(id, { rating: m.rating === v ? 0 : v });
        m.rating = m.rating === v ? 0 : v;
        renderSidebar();
    }
}

async function toggleStatus(m, value) {
    const next = m.status === value ? 'pending' : value;
    await postUpdate(m.stable_id, { status: next });
    m.status = next;
    renderSidebar();
}

async function setInOut(m, vIn, vOut) {
    const cur = effectiveSpan(m);
    const newIn = vIn != null ? vIn : cur[0];
    const newOut = vOut != null ? vOut : cur[1];
    if (newIn >= newOut) {
        toast('In must be before out');
        return;
    }
    const anchor = effectiveAnchor(m);
    const clampedAnchor = Math.max(newIn, Math.min(newOut, anchor));
    const body = { user_in_out: [newIn, newOut] };
    if (clampedAnchor !== anchor) body.user_anchor_override = clampedAnchor;
    await postUpdate(m.stable_id, body);
    m.user_in_out = [newIn, newOut];
    if (clampedAnchor !== anchor) m.user_anchor_override = clampedAnchor;
    renderSidebar();
}

async function setAnchor(m, t) {
    const [vIn, vOut] = effectiveSpan(m);
    const clamped = Math.max(vIn, Math.min(vOut, t));
    await postUpdate(m.stable_id, { user_anchor_override: clamped });
    m.user_anchor_override = clamped;
    renderSidebar();
}

async function postUpdate(id, body) {
    return fetch(`/api/moments/${encodeURIComponent(id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
}

async function addAtPlayhead() {
    const player = document.getElementById('player');
    const t = player.currentTime || 0;
    const ridePoint = window.findNearestPoint ? window.findNearestPoint(t) : null;
    const localRideT = ridePoint?.t ?? t;
    const rideT = (REVIEW.currentClipMeta?.ride_start_secs || 0) + localRideT;
    const body = {
        clip_name: REVIEW.currentClip,
        anchor_video_secs: t,
        video_start: Math.max(0, t - 5),
        video_end: t + 5,
        ride_time_secs: rideT,
        notes: 'manual add',
    };
    const r = await fetch('/api/moments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    const created = await r.json();
    const idx = REVIEW.moments.findIndex(m => m.stable_id === created.stable_id);
    if (idx >= 0) REVIEW.moments[idx] = created;
    else REVIEW.moments.push(created);
    REVIEW.activeMomentId = created.stable_id;
    renderAll();
}

function stepActiveMoment(direction) {
    if (!REVIEW.moments.length) return;
    const sorted = REVIEW.moments.slice().sort((a, b) => a.ride_time_secs - b.ride_time_secs);
    let idx = sorted.findIndex(m => m.stable_id === REVIEW.activeMomentId);
    if (idx === -1) idx = direction > 0 ? -1 : sorted.length;
    const next = sorted[Math.max(0, Math.min(sorted.length - 1, idx + direction))];
    if (!next) return;
    setActive(next.stable_id);
    seekTo(next.clip_name, effectiveAnchor(next));
}

function setActive(id) {
    REVIEW.activeMomentId = id;
    renderClipRibbon();
    // Update sidebar's is-active highlight without losing scroll position
    document.querySelectorAll('.moment').forEach(el => {
        el.classList.toggle('is-active', el.dataset.id === id);
    });
}

function seekTo(clipName, secs) {
    if (clipName !== REVIEW.currentClip) {
        switchClip(clipName, secs);
    } else {
        const player = document.getElementById('player');
        player.currentTime = secs;
        player.play().catch(() => {});
    }
}

function switchClip(filename, seekSecs = 0) {
    REVIEW.currentClip = filename;
    REVIEW.currentClipMeta = REVIEW.clips.find(c => c.filename === filename) || null;
    document.getElementById('clip-selector').value = filename;
    const player = document.getElementById('player');
    player.src = `/video/${encodeURIComponent(filename)}`;
    player.load();
    player.addEventListener('loadedmetadata', () => {
        if (seekSecs) player.currentTime = seekSecs;
        player.play().catch(() => {});
    }, { once: true });
    if (typeof window.reloadRideData === 'function') {
        window.reloadRideData(filename);
    }
    renderClipRibbon();
    renderRideStrip();
}

// ─── Helpers ───────────────────────────────────────────────────────

function effectiveAnchor(m) {
    return (m.user_anchor_override != null) ? m.user_anchor_override : m.anchor_video_secs;
}

function effectiveSpan(m) {
    return m.user_in_out != null
        ? [m.user_in_out[0], m.user_in_out[1]]
        : [m.video_start, m.video_end];
}

function formatTime(secs) {
    secs = Math.max(0, Math.round(secs));
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    return h
        ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
        : `${m}:${String(s).padStart(2, '0')}`;
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

function toast(msg) {
    const el = document.createElement('div');
    el.textContent = msg;
    Object.assign(el.style, {
        position: 'fixed', bottom: '20px', left: '50%', transform: 'translateX(-50%)',
        background: 'rgba(37, 99, 235, 0.95)', color: '#fff',
        padding: '8px 16px', borderRadius: '4px', fontSize: '13px',
        fontFamily: 'Rajdhani, sans-serif', zIndex: 9999,
        boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
    });
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2200);
}
