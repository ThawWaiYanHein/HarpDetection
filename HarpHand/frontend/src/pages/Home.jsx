import { Link } from 'react-router-dom';
import './Home.css';

export default function Home() {
  return (
    <div className="vintage-page">
      <div className="vintage-rivets vintage-rivets-left" aria-hidden />
      <div className="vintage-rivets vintage-rivets-right" aria-hidden />

      <header className="vintage-header">
        <div className="vintage-header-inner">
          <Link to="/" className="vintage-logo">NAT SHIN NAUNG</Link>
          <nav className="vintage-nav">
            <Link to="/">HOME</Link>
            <a href="#benefits">BENEFITS</a>
            <a href="#video">VIDEO</a>
            <a href="#contact">CONTACT</a>
          </nav>
          <Link to="/login" className="vintage-btn vintage-btn-cta">GET STARTED</Link>
        </div>
      </header>

      <main className="vintage-main">
        <section className="vintage-hero">
          <div className="vintage-hero-banner">
            <div className="vintage-hero-content">
              <h1 className="vintage-hero-title">Harp String Detection</h1>
              <p className="vintage-hero-desc">
                Capture which strings are plucked in your harp performance. Use audio detection, hand tracking, or both together—then export your note sheet or analysis.
              </p>
              <div className="vintage-hero-actions">
                <Link to="/login" className="vintage-btn vintage-btn-primary">GET STARTED</Link>
                <a href="#video" className="vintage-btn vintage-btn-secondary">WATCH THE VIDEO</a>
              </div>
            </div>
            <div className="vintage-hero-badge">
              <span className="vintage-badge-text">MYANMAR HARP</span>
            </div>
            <div className="vintage-chains" aria-hidden />
          </div>
        </section>

        <section id="benefits" className="vintage-benefits">
          <h2 className="vintage-section-title">Benefits</h2>
          <p className="vintage-section-subtitle">
            This tool helps musicians and teachers visualize and document harp string plucks from video—with optional hand and audio analysis.
          </p>
          <div className="vintage-benefits-grid">
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M24 8v8l6 6M18 14l6 6 12-12M12 28l6 6 12-12" /></svg>
              </div>
              <h3>Easy to use</h3>
              <p>Upload a video, choose audio and/or hand detection, and get timestamps and note sheets.</p>
            </div>
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 24h32M24 8v32M16 16l16 16M32 16L16 32" /></svg>
              </div>
              <h3>Audio + Hand</h3>
              <p>Combine onset detection with hand-tracking for higher accuracy and teaching insights.</p>
            </div>
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="24" cy="24" r="18" /><path d="M24 14v10l6 6" /></svg>
              </div>
              <h3>Precise</h3>
              <p>Frame-accurate timing and per-string statistics for practice and analysis.</p>
            </div>
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 12h24v24H12z" /><path d="M18 24l6 6 12-12" /></svg>
              </div>
              <h3>Export</h3>
              <p>Download CSV logs, annotated video, and PDF note sheets for your records.</p>
            </div>
          </div>
          <div className="vintage-gears vintage-gears-bottom" aria-hidden />
        </section>

        <section id="video" className="vintage-video">
          <h2 className="vintage-section-title">See it in action</h2>
          <p className="vintage-section-subtitle">Upload a harp performance video to run detection.</p>
          <Link to="/login" className="vintage-btn vintage-btn-primary">Try the tool</Link>
        </section>

        <section id="contact" className="vintage-contact">
          <h2 className="vintage-section-title">Contact</h2>
          <p className="vintage-section-subtitle">NAT SHIN NAUNG — Harp string detection for teaching and performance.</p>
        </section>
      </main>

      <footer className="vintage-footer">
        <p>NAT SHIN NAUNG · Harp String Detection · Audio · Hand · Both</p>
      </footer>
    </div>
  );
}
