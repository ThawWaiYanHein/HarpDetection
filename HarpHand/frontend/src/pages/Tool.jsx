import { useState, useRef, useCallback, useEffect } from 'react'
import { Link } from 'react-router-dom'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'
import '../App.css'

const API = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  const [modelFile, setModelFile] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [method, setMethod] = useState('audio')  // 'audio' | 'hand' | 'both'
  const [mode, setMode] = useState('hybrid')   // 'default' | 'hybrid' (audio only)
  const [weightsFile, setWeightsFile] = useState(null)
  const [logs, setLogs] = useState([])
  const [videoUrl, setVideoUrl] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [logViewMode, setLogViewMode] = useState('list')  // 'list' | 'grid'
  const [videoDuration, setVideoDuration] = useState(0)
  const [useDefaultModel, setUseDefaultModel] = useState(false)
  const [useDefaultWeights, setUseDefaultWeights] = useState(true)
  const [defaultsAvailable, setDefaultsAvailable] = useState({ default_model: false, default_weights: false })
  const videoRef = useRef(null)
  const generatedNoteRef = useRef(null)
  const logPanelRef = useRef(null)
  const modelInput = useRef(null)
  const videoInput = useRef(null)
  const weightsInput = useRef(null)

  useEffect(() => {
    fetch(`${API}/defaults`)
      .then((r) => {
        if (!r.ok) throw new Error('API not ok')
        return r.json()
      })
      .then((d) => {
        setDefaultsAvailable(d)
        setUseDefaultModel(!!d.default_model)
        setUseDefaultWeights(!!d.default_weights)
      })
      .catch(() => {
        setDefaultsAvailable({ default_model: false, default_weights: false })
        setUseDefaultModel(false)
        setUseDefaultWeights(false)
      })
  }, [])

  // Logs visible only within this window (seconds) of current video time
  const LOG_TIME_WINDOW = 0.25
  const visibleLogs = logs.filter(
    (e) => Math.abs((e.time || 0) - currentTime) <= LOG_TIME_WINDOW
  )

  // Group by time for grid view: one row per pluck, with string + hand + match
  const gridRows = (() => {
    const byTime = {}
    logs.forEach((e) => {
      const t = e.time ?? 0
      if (!byTime[t]) byTime[t] = { time: t, audio: [], hand: [] }
      if (e.type === 'audio') byTime[t].audio.push(e)
      else if (e.type === 'hand') byTime[t].hand.push(e)
    })
    return Object.values(byTime)
      .sort((a, b) => a.time - b.time)
      .map((row, idx) => {
        const audioStrings = [...new Set(row.audio.map((e) => e.string))]
        const handStrings = [...new Set(row.hand.map((e) => e.string))]
        const match = audioStrings.some((s) => handStrings.includes(s))
        const stringMain = audioStrings.length ? audioStrings.join(', ') : '-'
        const handMain = handStrings.length ? handStrings.join(', ') : '-'
        const note = row.audio[0]
          ? row.audio.map((a) => `${(a.confidence * 100).toFixed(0)}%`).join(', ')
          : row.hand[0]
            ? `${row.hand[0].finger || ''} ${(row.hand[0].confidence * 100).toFixed(0)}%`.trim()
            : ''
        return {
          index: idx + 1,
          time: row.time,
          stringMain,
          handMain,
          match,
          note,
          inWindow: Math.abs(row.time - currentTime) <= LOG_TIME_WINDOW,
          audio: row.audio,
          hand: row.hand,
        }
      })
  })()

  // Per-string stats (when we have both audio and hand): for each string, total plucks and matches
  const perStringStats = (() => {
    const stats = {}
    for (let s = 1; s <= 16; s++) stats[`S${s}`] = { total: 0, matches: 0 }
    gridRows.forEach((row) => {
      const audioStrs = (row.stringMain || '').split(',').map((x) => x.trim()).filter(Boolean)
      audioStrs.forEach((s) => {
        if (stats[s] != null) {
          stats[s].total += 1
          if (row.match) stats[s].matches += 1
        }
      })
    })
    return stats
  })()

  // Agreement matrix: count of (audio, hand) pairs per pluck (when they match we count the pair)
  const agreementMatrix = (() => {
    const m = {}
    for (let i = 1; i <= 16; i++) {
      m[`S${i}`] = {}
      for (let j = 1; j <= 16; j++) m[`S${i}`][`S${j}`] = 0
    }
    gridRows.forEach((row) => {
      const audioStrs = (row.stringMain || '').split(',').map((x) => x.trim()).filter(Boolean)
      const handStrs = (row.handMain || '').split(',').map((x) => x.trim()).filter(Boolean)
      audioStrs.forEach((a) => {
        handStrs.forEach((h) => {
          if (m[a]?.[h] != null) m[a][h] += 1
        })
      })
    })
    return m
  })()

  const pollStatus = useCallback(async (id) => {
    const res = await fetch(`${API}/status/${id}`)
    if (!res.ok) throw new Error('Status check failed')
    const data = await res.json()
    setStatus(data)
    if (data.status === 'done') {
      // Fetch logs and video URL
      try {
        const logsRes = await fetch(`${API}/logs/${id}`)
        if (logsRes.ok) {
          const logsData = await logsRes.json()
          setLogs(logsData.events || [])
        }
        // Get video URL (prefer combined, fallback to audio/hand, or use direct path for single mode)
        let videoUrlToSet = null
        if (data.audio && data.hand) {
          const type = data.combined ? 'combined' : 'audio'
          videoUrlToSet = `${API}/video-stream/${id}?type=${type}`
        } else if (data.video_path) {
          videoUrlToSet = `${API}/video-stream/${id}`
        }
        if (videoUrlToSet) {
          setVideoUrl(videoUrlToSet)
        }
      } catch (err) {
        console.error('Error fetching logs/video:', err)
      }
      return
    }
    if (data.status === 'error') return
    setTimeout(() => pollStatus(id), 1500)
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!videoFile) {
      setError('Please select a video file.')
      return
    }
    if ((method === 'audio' || method === 'both') && !useDefaultModel && !modelFile) {
      setError('Use default model or select a .keras model file.')
      return
    }
    if ((method === 'audio' || method === 'both') && useDefaultModel && !defaultsAvailable.default_model) {
      setError('Default model is not available. Place default.keras in backend/models/ or upload your own.')
      return
    }
    if ((method === 'hand' || method === 'both') && !useDefaultWeights && !weightsFile) {
      setError('Use default weights or select a .pt weights file.')
      return
    }
    if ((method === 'hand' || method === 'both') && useDefaultWeights && !defaultsAvailable.default_weights) {
      setError('Default weights are not available. Place best.pt in backend/weights/ or upload your own.')
      return
    }
    setError(null)
    setStatus(null)
    setUploading(true)
    try {
      const form = new FormData()
      form.append('method', method)
      form.append('video', videoFile)
      if (method === 'audio' || method === 'both') {
        const sendDefaultModel = defaultsAvailable.default_model ? useDefaultModel : false;
        form.append('use_default_model', sendDefaultModel ? 'true' : 'false')
        form.append('mode', mode)
        if (!sendDefaultModel && modelFile) form.append('model', modelFile)
      }
      if (method === 'hand' || method === 'both') {
        const sendDefaultWeights = defaultsAvailable.default_weights ? useDefaultWeights : false;
        form.append('use_default_weights', sendDefaultWeights ? 'true' : 'false')
        if (!sendDefaultWeights && weightsFile) form.append('weights', weightsFile)
      }
      const res = await fetch(`${API}/upload`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const { job_id } = await res.json()
      setJobId(job_id)
      pollStatus(job_id)
    } catch (err) {
      setError(err.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const reset = () => {
    setModelFile(null)
    setVideoFile(null)
    setWeightsFile(null)
    setJobId(null)
    setStatus(null)
    setError(null)
    setLogs([])
    setVideoUrl(null)
    setCurrentTime(0)
    setVideoDuration(0)
    if (modelInput.current) modelInput.current.value = ''
    if (videoInput.current) videoInput.current.value = ''
    if (weightsInput.current) weightsInput.current.value = ''
  }

  const handleDownloadNotePdf = async () => {
    if (!generatedNoteRef.current || !noteRows.length) return
    try {
      const node = generatedNoteRef.current
      // Temporarily expand the container so html2canvas captures the full grid
      const origMaxHeight = node.style.maxHeight
      const origOverflow = node.style.overflow
      node.style.maxHeight = 'none'
      node.style.overflow = 'visible'

      const canvas = await html2canvas(node, {
        scale: window.devicePixelRatio || 2,
        useCORS: true,
        backgroundColor: '#f5f0e1',
        scrollY: -window.scrollY,
        windowHeight: node.scrollHeight + 200,
      })

      // Restore original styles
      node.style.maxHeight = origMaxHeight
      node.style.overflow = origOverflow

      const imgData = canvas.toDataURL('image/png')
      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'pt',
        format: 'a4',
      })
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()

      // Mode label
      const modeLabel = method === 'both' ? 'Both (Audio + Hand)' : method === 'audio' ? 'Audio Detection' : 'Hand Detection'

      // Accuracy info
      const totalEvents = gridRows.length
      const matchCount = gridRows.filter((r) => r.match).length
      const accuracyPct = totalEvents > 0 ? ((matchCount / totalEvents) * 100).toFixed(1) : '0.0'
      const isBoth = !!(status?.audio && status?.hand)

      // Title
      pdf.setFontSize(16)
      pdf.text(modeLabel, pageWidth / 2, 30, { align: 'center' })

      // Stats line
      pdf.setFontSize(10)
      let statsText = `${totalEvents} events`
      if (isBoth) {
        statsText += ` · ${matchCount} matches · ${accuracyPct}% accuracy`
      } else {
        // Single mode: show average confidence
        const confidences = logs.filter(e => typeof e.confidence === 'number').map(e => e.confidence)
        if (confidences.length > 0) {
          const avgConf = ((confidences.reduce((a, b) => a + b, 0) / confidences.length) * 100).toFixed(1)
          statsText += ` · ${avgConf}% avg confidence`
        }
      }
      pdf.text(statsText, pageWidth / 2, 48, { align: 'center' })

      // Calculate image size to fit the page
      const imgWidth = pageWidth - 80
      const imgHeight = (canvas.height * imgWidth) / canvas.width

      // If image is too tall for one page, scale it down or split across pages
      const availableHeight = pageHeight - 70
      if (imgHeight <= availableHeight) {
        const offsetY = 60
        pdf.addImage(imgData, 'PNG', 40, offsetY, imgWidth, imgHeight, undefined, 'FAST')
      } else {
        // Scale to fit width, then paginate
        const scaledWidth = imgWidth
        const scaledHeight = imgHeight
        let yOffset = 0
        let pageNum = 0
        while (yOffset < scaledHeight) {
          if (pageNum > 0) pdf.addPage()
          const sourceY = (yOffset / scaledHeight) * canvas.height
          const sourceH = Math.min((availableHeight / scaledHeight) * canvas.height, canvas.height - sourceY)
          const sliceCanvas = document.createElement('canvas')
          sliceCanvas.width = canvas.width
          sliceCanvas.height = sourceH
          const ctx = sliceCanvas.getContext('2d')
          ctx.drawImage(canvas, 0, sourceY, canvas.width, sourceH, 0, 0, canvas.width, sourceH)
          const sliceData = sliceCanvas.toDataURL('image/png')
          const sliceH = (sourceH * scaledWidth) / canvas.width
          pdf.addImage(sliceData, 'PNG', 40, pageNum === 0 ? 60 : 30, scaledWidth, sliceH, undefined, 'FAST')
          yOffset += availableHeight
          pageNum++
        }
      }
      pdf.save(`${method}-detection-note.pdf`)
    } catch (err) {
      console.error('Failed to generate PDF', err)
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(2)
    return `${String(mins).padStart(2, '0')}:${secs.padStart(5, '0')}`
  }

  const downloadLog = () => {
    const headers = ['#', 'Time', 'Type', 'String', 'Finger', 'Confidence%', 'Method', 'Distance(px)', 'Status']
    const rows = logs.map((e, i) => {
      const num = e.entry_number ?? String(i + 1).padStart(4, '0')
      const time = formatTime(e.time ?? 0)
      const type = e.type === 'audio' ? 'string' : 'hand'
      const str = e.string ?? ''
      const finger = e.finger ?? ''
      const conf = e.confidence != null ? (e.confidence * 100).toFixed(1) : ''
      const method = e.method ?? ''
      const dist = e.distance != null ? e.distance.toFixed(1) : ''
      const status = e.status ?? ''
      return [num, time, type, str, finger, conf, method, dist, status]
    })
    const csv = [headers.join(','), ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = jobId ? `detection_log_${jobId}.csv` : 'detection_log.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadCsv = (type) => {
    if (!jobId) return
    const url = type ? `${API}/download/csv/${jobId}?type=${type}` : `${API}/download/csv/${jobId}`
    window.open(url, '_blank')
  }

  const downloadVideo = (type) => {
    if (!jobId) return
    const url = type ? `${API}/download/video/${jobId}?type=${type}` : `${API}/download/video/${jobId}`
    window.open(url, '_blank')
  }

  const seekToTime = (timeInSeconds) => {
    if (videoRef.current != null && !isNaN(timeInSeconds)) {
      videoRef.current.currentTime = timeInSeconds
      setCurrentTime(timeInSeconds)
    }
  }

  const goToNextPluck = () => {
    const next = gridRows.find((r) => r.time > currentTime)
    if (next) seekToTime(next.time)
  }

  const goToPrevPluck = () => {
    const prev = [...gridRows].reverse().find((r) => r.time < currentTime)
    if (prev) seekToTime(prev.time)
  }

  const hasBothAudioHand = gridRows.some((r) => r.stringMain !== '-' && r.handMain !== '-')

  // Generated Note: one cell per sound event; 8 columns, unlimited rows
  const NOTE_COLS = 8
  const noteCells = gridRows.map((row) => {
    const baseStr = row.stringMain !== '-' ? row.stringMain : row.handMain
    if (baseStr === '-') return { parts: [], together: false }

    // Map which strings are plucked with thumb for this event
    const thumbStrings = new Set(
      (row.hand || [])
        .filter((h) => String(h.finger || '').toLowerCase() === 'thumb')
        .map((h) => String(h.string || '').trim())
    )

    const rawParts = (baseStr || '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)

    const parts =
      rawParts
        .map((s) => {
          const num = s.replace(/^S\s*/i, '').replace(/\D/g, '')
          if (!num) return null
          const canonical = s.toUpperCase().startsWith('S') ? s : `S${num}`
          const isThumb = thumbStrings.has(canonical)
          return { num, thumb: isThumb }
        })
        .filter(Boolean) || []

    return { parts, together: parts.length > 1 }
  })
  const noteRows = []
  for (let i = 0; i < noteCells.length; i += NOTE_COLS) {
    noteRows.push(noteCells.slice(i, i + NOTE_COLS))
  }

  // User profile from localStorage
  const userName = localStorage.getItem('user_name') || ''
  const userAvatar = localStorage.getItem('user_avatar') || ''
  const userEmail = localStorage.getItem('user_email') || ''
  const userInitials = userName
    ? userName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : (userEmail ? userEmail[0].toUpperCase() : 'U')

  return (
    <div className="app">
      <header className="header header-vintage" style={{ position: 'relative' }}>
        <div className="header-vintage-brand">
          <Link to="/" className="header-vintage-logo">NAT SHIN NAUNG</Link>
          <h1>Harp String Detection</h1>
          <p className="tagline">Audio, hand, or both on the same video</p>
        </div>
        {userName && (
          <div className="user-profile-badge" title={userName + (userEmail ? `\n${userEmail}` : '')}>
            {userAvatar ? (
              <img src={userAvatar} alt={userName} className="user-profile-avatar" referrerPolicy="no-referrer" />
            ) : (
              <span className="user-profile-initials">{userInitials}</span>
            )}
            <span className="user-profile-name">{userName.split(' ')[0]}</span>
          </div>
        )}
      </header>

      <main className="main">
        <section className="card upload-card">
          <form onSubmit={handleSubmit} className="upload-form">
            <div className="field">
              <label>Detection type</label>
              <div className="radio-group">
                <label className="radio">
                  <input
                    type="radio"
                    name="method"
                    value="audio"
                    checked={method === 'audio'}
                    onChange={() => setMethod('audio')}
                  />
                  <span>Audio (model + onset detection)</span>
                </label>
                <label className="radio">
                  <input
                    type="radio"
                    name="method"
                    value="hand"
                    checked={method === 'hand'}
                    onChange={() => setMethod('hand')}
                  />
                  <span>Hand (YOLO strings + MediaPipe touch)</span>
                </label>
                <label className="radio">
                  <input
                    type="radio"
                    name="method"
                    value="both"
                    checked={method === 'both'}
                    onChange={() => setMethod('both')}
                  />
                  <span>Both (audio + hand on same video)</span>
                </label>
              </div>
            </div>

            {(method === 'audio' || method === 'both') && (
              <div className="field">
                <label>Audio model</label>
                {defaultsAvailable.default_model ? (
                  <div className="radio-group">
                    <label className="radio">
                      <input
                        type="radio"
                        name="modelSource"
                        checked={useDefaultModel}
                        onChange={() => { setUseDefaultModel(true); setModelFile(null) }}
                      />
                      <span>Use default model (app)</span>
                    </label>
                    <label className="radio">
                      <input
                        type="radio"
                        name="modelSource"
                        checked={!useDefaultModel}
                        onChange={() => setUseDefaultModel(false)}
                      />
                      <span>Upload my model</span>
                    </label>
                  </div>
                ) : (
                  <p className="field-hint">Place default.keras in backend/models/ to use default, or upload your model below.</p>
                )}
                {(!useDefaultModel || !defaultsAvailable.default_model) && (
                  <input
                    id="model"
                    ref={modelInput}
                    type="file"
                    accept=".keras"
                    onChange={(e) => {
                      setModelFile(e.target.files?.[0] ?? null)
                      if (e.target.files?.[0]) setUseDefaultModel(false)
                    }}
                    className="mt-1"
                  />
                )}
              </div>
            )}

            {(method === 'hand' || method === 'both') && (
              <div className="field">
                <label>String detection model</label>
                {defaultsAvailable.default_weights ? (
                  <div className="radio-group">
                    <label className="radio">
                      <input
                        type="radio"
                        name="weightsSource"
                        checked={useDefaultWeights}
                        onChange={() => { setUseDefaultWeights(true); setWeightsFile(null) }}
                      />
                      <span>Use default weights (app)</span>
                    </label>
                    <label className="radio">
                      <input
                        type="radio"
                        name="weightsSource"
                        checked={!useDefaultWeights}
                        onChange={() => setUseDefaultWeights(false)}
                      />
                      <span>Upload my weights</span>
                    </label>
                  </div>
                ) : (
                  <p className="field-hint">Place best.pt in backend/weights/ to use default, or upload your weights below.</p>
                )}
                {(!useDefaultWeights || !defaultsAvailable.default_weights) && (
                  <input
                    id="weights"
                    ref={weightsInput}
                    type="file"
                    accept=".pt"
                    onChange={(e) => {
                      setWeightsFile(e.target.files?.[0] ?? null)
                      if (e.target.files?.[0]) setUseDefaultWeights(false)
                    }}
                    className="mt-1"
                  />
                )}
              </div>
            )}

            <div className="field">
              <label htmlFor="video">Video (.mp4, .mov, .mkv, .avi, .webm)</label>
              <input
                id="video"
                ref={videoInput}
                type="file"
                accept=".mp4,.mov,.mkv,.avi,.webm"
                onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
              />
            </div>

            {(method === 'audio' || method === 'both') && (
              <div className="field">
                <label>Audio mode</label>
                <div className="radio-group">
                  <label className="radio">
                    <input
                      type="radio"
                      name="mode"
                      value="default"
                      checked={mode === 'default'}
                      onChange={() => setMode('default')}
                    />
                    <span>Default (model only, threshold 0.25)</span>
                  </label>
                  <label className="radio">
                    <input
                      type="radio"
                      name="mode"
                      value="hybrid"
                      checked={mode === 'hybrid'}
                      onChange={() => setMode('hybrid')}
                    />
                    <span>Hybrid (model + YIN fallback)</span>
                  </label>
                </div>
              </div>
            )}
            {error && <p className="error">{error}</p>}
            <button type="submit" className="btn primary" disabled={uploading}>
              {uploading ? 'Uploading…' : 'Run detection'}
            </button>
          </form>
        </section>

        {status && (
          <section className="card status-card">
            <h3>Status</h3>
            {status.status === 'queued' && <p className="muted">Queued…</p>}
            {status.status === 'running' && (
              <p className="running">{status.message || 'Processing…'}</p>
            )}
            {status.status === 'error' && (
              <p className="error">{status.message}</p>
            )}
            {status.status === 'done' && (
              <div className="done">
                {status.audio && status.hand ? (
                  <>
                    <p className="success">
                      Done. Audio: {status.audio.rows ?? 0} onset(s). Hand: {status.hand.rows ?? 0} touch(es).
                    </p>
                    {status.combined ? (
                      <div className="actions" style={{ marginBottom: '0.5rem' }}>
                        <button type="button" className="btn primary" onClick={() => downloadVideo('combined')} style={{ fontSize: '1rem', padding: '0.75rem 1.5rem' }}>
                          📹 Combined Video (hand + audio)
                        </button>
                      </div>
                    ) : status.combined_error && (
                      <p className="error" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                        Combined video: {status.combined_error}
                      </p>
                    )}
                    <div className="actions actions-both">
                      <button type="button" className="btn primary" onClick={() => downloadCsv('audio')}>
                        Audio CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo('audio')}>
                        Audio video
                      </button>
                      <button type="button" className="btn primary" onClick={() => downloadCsv('hand')}>
                        Hand CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo('hand')}>
                        Hand video
                      </button>
                    </div>
                    <div className="actions" style={{ marginTop: '0.5rem' }}>
                      <button type="button" className="btn primary" onClick={downloadLog}>
                        📥 Detection log (CSV)
                      </button>
                    </div>
                  </>
                ) : status.audio ? (
                  <>
                    <p className="success">
                      Done (audio only). {status.audio.rows ?? 0} onset(s).
                    </p>
                    {status.hand_error && (
                      <p className="error" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                        Hand detection: {status.hand_error}
                      </p>
                    )}
                    <div className="actions">
                      <button type="button" className="btn primary" onClick={() => downloadCsv('audio')}>
                        Download CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo('audio')}>
                        Download video
                      </button>
                    </div>
                    <div className="actions" style={{ marginTop: '0.5rem' }}>
                      <button type="button" className="btn primary" onClick={downloadLog}>
                        📥 Detection log (CSV)
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="success">
                      Done. Detected {status.rows ?? 0} {method === 'hand' ? 'touch event(s)' : 'onset(s)'}.
                    </p>
                    <div className="actions">
                      <button type="button" className="btn primary" onClick={() => downloadCsv()}>
                        Download CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo()}>
                        Download {method === 'hand' ? 'annotated' : 'labeled'} video
                      </button>
                    </div>
                    <div className="actions" style={{ marginTop: '0.5rem' }}>
                      <button type="button" className="btn primary" onClick={downloadLog}>
                        📥 Detection log (CSV)
                      </button>
                    </div>
                  </>
                )}
                <div className="actions" style={{ marginTop: '0.5rem' }}>
                  <button type="button" className="btn ghost" onClick={reset}>
                    New run
                  </button>
                </div>
              </div>
            )}
          </section>
        )}

        {status?.status === 'done' && videoUrl && (
          <section className="card preview-card">
            <h3>Preview</h3>
            {gridRows.length > 0 && (
              <div className="preview-nav">
                <button type="button" className="btn ghost" onClick={goToPrevPluck} title="Previous pluck">
                  ← Prev pluck
                </button>
                <button type="button" className="btn ghost" onClick={goToNextPluck} title="Next pluck">
                  Next pluck →
                </button>
              </div>
            )}
            <div className="video-container">
              <video
                ref={videoRef}
                controls
                src={videoUrl}
                className="preview-video"
                onLoadedMetadata={() => {
                  if (videoRef.current && isFinite(videoRef.current.duration))
                    setVideoDuration(videoRef.current.duration)
                }}
                onTimeUpdate={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onSeeked={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onPlay={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
            {gridRows.length > 0 && videoDuration > 0 && (
              <div className="timeline-strip" title="Click to seek">
                <div
                  className="timeline-playhead"
                  style={{ left: `${(currentTime / videoDuration) * 100}%` }}
                />
                {gridRows.map((row) => (
                  <button
                    key={`tl-${row.time}-${row.index}`}
                    type="button"
                    className={`timeline-marker ${row.match ? 'timeline-marker-match' : 'timeline-marker-miss'}`}
                    style={{ left: `${(row.time / videoDuration) * 100}%` }}
                    onClick={() => seekToTime(row.time)}
                    title={`${formatTime(row.time)} ${row.match ? '✓' : ''}`}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {status?.status === 'done' && logs.length > 0 && (
          <section className="card log-card">
            <div className="log-card-header">
              <h3>Detection Log (synced to video · {formatTime(currentTime)})</h3>
              <div className="log-view-toggle">
                <button
                  type="button"
                  className={`btn ghost ${logViewMode === 'list' ? 'active' : ''}`}
                  onClick={() => setLogViewMode('list')}
                >
                  List
                </button>
                <button
                  type="button"
                  className={`btn ghost ${logViewMode === 'grid' ? 'active' : ''}`}
                  onClick={() => setLogViewMode('grid')}
                >
                  Grid
                </button>
              </div>
            </div>
            <div className="log-download-row">
              <span className="log-summary">
                {gridRows.length} events
                {(status?.audio && status?.hand) ? (
                  <> · {gridRows.filter((r) => r.match).length} matches · {gridRows.length > 0 ? ((gridRows.filter((r) => r.match).length / gridRows.length) * 100).toFixed(1) : '0'}% accuracy</>
                ) : (() => {
                  const confs = logs.filter(e => typeof e.confidence === 'number').map(e => e.confidence)
                  const avg = confs.length > 0 ? ((confs.reduce((a, b) => a + b, 0) / confs.length) * 100).toFixed(1) : null
                  return avg ? <> · {avg}% avg confidence</> : null
                })()}
              </span>
              <button type="button" className="btn primary" onClick={downloadLog} title="Download full detection log as CSV">
                Download log (CSV)
              </button>
            </div>
            {logViewMode === 'list' ? (
              <div ref={logPanelRef} className="log-panel">
                {visibleLogs.length === 0 ? (
                  <p className="muted" style={{ padding: '1rem', textAlign: 'center' }}>
                    No events at this time. Play the video to see detections.
                  </p>
                ) : (
                  visibleLogs.map((event, idx) => (
                    <div
                      key={event.entry_number || idx}
                      role="button"
                      tabIndex={0}
                      className="log-entry log-entry-active log-entry-clickable"
                      onClick={() => seekToTime(event.time)}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seekToTime(event.time) } }}
                      title="Click to seek video to this time"
                    >
                      <span className="log-entry-number">{event.entry_number || `${(idx + 1).toString().padStart(4, '0')}`}</span>
                      <span className="log-time">{formatTime(event.time)}</span>
                      <span className={`log-type log-type-${event.type}`}>
                        {event.type === 'audio' ? 'string' : 'hand'}
                      </span>
                      <span className="log-string">{event.string}</span>
                      {event.type === 'audio' && (
                        <span style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                          <span className="log-method">{event.method === 'yin' || event.method === 'string*' ? 'String*' : (event.method === 'model' || event.method === 'string' ? 'String' : (event.method || 'String'))}</span>
                          <span className="log-confidence">{(event.confidence * 100).toFixed(1)}%</span>
                        </span>
                      )}
                      {event.type === 'hand' && (
                        <span style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                          <span className="log-status">{event.status || 'detected'}</span>
                          <span className="log-finger">{event.finger || '-'}</span>
                          <span className="log-distance">{event.distance ? event.distance.toFixed(1) : '0.0'}px</span>
                          <span className="log-confidence">{(event.confidence * 100).toFixed(1)}%</span>
                        </span>
                      )}
                    </div>
                  ))
                )}
              </div>
            ) : (
              <div ref={logPanelRef} className="log-panel log-grid-wrap">
                <table className="log-grid">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Time</th>
                      <th>String</th>
                      <th>Hand</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {gridRows.map((row) => (
                      <tr
                        key={`${row.time}-${row.index}`}
                        role="button"
                        tabIndex={0}
                        className={`log-grid-row-clickable ${row.inWindow ? 'log-grid-row-active' : ''}`}
                        onClick={() => seekToTime(row.time)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seekToTime(row.time) } }}
                        title="Click to seek video to this time"
                      >
                        <td className="log-grid-num">{row.index}</td>
                        <td className="log-grid-time">{formatTime(row.time)}</td>
                        <td className="log-grid-cell">
                          <span className="log-grid-main">{row.stringMain}</span>
                          {row.note ? <span className="log-grid-annot">{row.note}</span> : null}
                        </td>
                        <td className={`log-grid-cell ${row.match ? 'log-grid-cell-match' : ''}`}>
                          <span className="log-grid-main">{row.handMain}</span>
                        </td>
                        <td className="log-grid-note">{row.match ? '✓' : ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {status?.status === 'done' && logs.length > 0 && (
          <section className="card generated-note-card">
            <div className="generated-note-header-row">
              <div>
                <h3 style={{ margin: 0 }}>
                  {method === 'both' ? 'Both (Audio + Hand)' : method === 'audio' ? 'Audio Detection' : 'Hand Detection'}
                  {' '}
                  — Note
                </h3>
                <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                  {gridRows.length} events
                  {(status?.audio && status?.hand) ? (
                    <> · {gridRows.filter((r) => r.match).length} matches · {gridRows.length > 0 ? ((gridRows.filter((r) => r.match).length / gridRows.length) * 100).toFixed(1) : '0'}% accuracy</>
                  ) : (() => {
                    const confs = logs.filter(e => typeof e.confidence === 'number').map(e => e.confidence)
                    const avg = confs.length > 0 ? ((confs.reduce((a, b) => a + b, 0) / confs.length) * 100).toFixed(1) : null
                    return avg ? <> · {avg}% avg confidence</> : null
                  })()}
                </span>
              </div>
              {noteRows.length > 0 && (
                <button type="button" className="btn secondary btn-sm" onClick={handleDownloadNotePdf}>
                  Download as PDF
                </button>
              )}
            </div>
            <p className="generated-note-desc">Each box is one sound event. Single string = number only; strings plucked together = numbers with underline.</p>
            {noteRows.length > 0 ? (
              <div className="generated-note-grid-wrap" ref={generatedNoteRef}>
                <div className="generated-note-grid" style={{ gridTemplateColumns: `repeat(${NOTE_COLS}, 1fr)` }}>
                  {noteRows.map((row, ri) =>
                    row.map((cell, ci) => {
                      const flatIndex = ri * NOTE_COLS + ci
                      const eventTime = gridRows[flatIndex]?.time
                      const hasParts = cell.parts && cell.parts.length > 0
                      return (
                        <div
                          key={`${ri}-${ci}`}
                          className={`generated-note-cell${hasParts ? ' generated-note-cell-clickable' : ''}`}
                          role={hasParts ? 'button' : undefined}
                          tabIndex={hasParts ? 0 : -1}
                          onClick={() => hasParts && eventTime != null && seekToTime(eventTime)}
                          onKeyDown={(e) => {
                            if (!hasParts) return
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              if (eventTime != null) seekToTime(eventTime)
                            }
                          }}
                          title={hasParts && eventTime != null ? `Seek to ${formatTime(eventTime)}` : ''}
                        >
                          {hasParts && (
                            <span className={cell.together ? 'generated-note-together' : ''}>
                              {cell.parts.map((p, idx) => (
                                <span key={idx} className="generated-note-token">
                                  <span
                                    className={
                                      'generated-note-thumb-dot' +
                                      (p.thumb ? '' : ' generated-note-thumb-dot--placeholder')
                                    }
                                    aria-hidden
                                  >
                                    ·
                                  </span>
                                  <span>{p.num}</span>
                                </span>
                              ))}
                            </span>
                          )}
                        </div>
                      )
                    })
                  )}
                </div>
              </div>
            ) : (
              <p className="muted">No events to show.</p>
            )}
          </section>
        )}

        {status?.status === 'done' && logs.length > 0 && hasBothAudioHand && (
          <section className="card analysis-card">
            <h3>Analysis</h3>
            <div className="analysis-section">
              <h4 className="analysis-subtitle">Per-string match rate</h4>
              <div className="per-string-table-wrap">
                <table className="per-string-table">
                  <thead>
                    <tr>
                      <th>String</th>
                      <th>Plucks</th>
                      <th>Matches</th>
                      <th>%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((s) => {
                      const key = `S${s}`
                      const st = perStringStats[key]
                      if (!st || st.total === 0) return null
                      const pct = st.total > 0 ? ((st.matches / st.total) * 100).toFixed(0) : '0'
                      return (
                        <tr key={key}>
                          <td className="per-string-name">{key}</td>
                          <td>{st.total}</td>
                          <td>{st.matches}</td>
                          <td className="per-string-pct">{pct}%</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="analysis-section">
              <h4 className="analysis-subtitle">Audio vs hand (agreement)</h4>
              <p className="muted analysis-hint">Count of times each (audio string, hand string) pair occurred.</p>
              <div className="agreement-matrix-wrap">
                <table className="agreement-matrix">
                  <thead>
                    <tr>
                      <th></th>
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((j) => (
                        <th key={j}>S{j}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((i) => {
                      const row = agreementMatrix[`S${i}`]
                      const hasAny = row && Object.values(row).some((v) => v > 0)
                      if (!hasAny) return null
                      return (
                        <tr key={`row-S${i}`}>
                          <th>S{i}</th>
                          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((j) => {
                            const v = row[`S${j}`] || 0
                            return (
                              <td key={j} className={v > 0 ? 'agreement-cell' : ''} title={`Audio S${i} / Hand S${j}`}>
                                {v > 0 ? v : ''}
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>NAT SHIN NAUNG · Audio · Hand · Both · For teaching / presentation</p>
      </footer>
    </div>
  )
}
