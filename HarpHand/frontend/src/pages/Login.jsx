import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';
import './Login.css';
import harp1 from '../assets/myanmar_harp.jpg';
import harp2 from '../assets/harp_2.jpg';
import harp3 from '../assets/harp_3.jpg';

const CAROUSEL_IMAGES = [harp1, harp2, harp3];
const CAROUSEL_INTERVAL = 5000; // 5 seconds

export default function Login() {
    const navigate = useNavigate();

    const [authError, setAuthError] = useState(null);
    const [isAuthenticating, setIsAuthenticating] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [currentSlide, setCurrentSlide] = useState(0);
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');

    // Auto-rotate carousel
    useEffect(() => {
        const timer = setInterval(() => {
            setCurrentSlide((prev) => (prev + 1) % CAROUSEL_IMAGES.length);
        }, CAROUSEL_INTERVAL);
        return () => clearInterval(timer);
    }, []);

    const handleLogin = (e) => {
        e.preventDefault();
        // Save user display name to localStorage
        const displayName = `${firstName} ${lastName}`.trim();
        if (displayName) {
            localStorage.setItem('user_name', displayName);
            localStorage.setItem('user_avatar', '');
            localStorage.setItem('user_email', '');
        }
        navigate('/tool');
    };

    const loginWithGoogle = useGoogleLogin({
        flow: 'auth-code',
        onSuccess: async (codeResponse) => {
            try {
                setIsAuthenticating(true);
                setAuthError(null);
                const API = import.meta.env.VITE_API_URL || '/api';
                const res = await fetch(`${API}/auth/google`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ token: codeResponse.code }),
                });

                if (!res.ok) throw new Error('Failed to authenticate with backend');

                const data = await res.json();
                // Save Google profile info to localStorage
                if (data.name) localStorage.setItem('user_name', data.name);
                if (data.picture) localStorage.setItem('user_avatar', data.picture);
                if (data.email) localStorage.setItem('user_email', data.email);

                navigate('/tool');
            } catch (err) {
                setAuthError('Google sign-in failed. Please try again.');
                console.error(err);
            } finally {
                setIsAuthenticating(false);
            }
        },
        onError: () => {
            setAuthError('Google sign-in was canceled or failed.');
        }
    });

    return (
        <div className="login-container">
            {/* Left split: Image Carousel */}
            <div className="login-image-side">
                {CAROUSEL_IMAGES.map((src, idx) => (
                    <img
                        key={idx}
                        src={src}
                        alt={`Myanmar Harp ${idx + 1}`}
                        className={`login-background-image ${idx === currentSlide ? 'slide-active' : 'slide-hidden'}`}
                    />
                ))}
                <div className="login-image-overlay">
                    <Link to="/" className="login-logo">NAT SHIN NAUNG</Link>
                    <div className="login-image-text">
                        <h2>Capturing Moments,</h2>
                        <h2>Creating Memories</h2>
                        <div className="carousel-indicators">
                            {CAROUSEL_IMAGES.map((_, idx) => (
                                <button
                                    key={idx}
                                    type="button"
                                    className={`indicator ${idx === currentSlide ? 'active' : ''}`}
                                    onClick={() => setCurrentSlide(idx)}
                                    aria-label={`Go to slide ${idx + 1}`}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Right split: Form */}
            <div className="login-form-side">
                <div className="login-form-wrapper">
                    <h1 className="login-title">Create an account</h1>
                    <p className="login-subtitle">
                        Already have an account? <a href="#" className="login-link">Log in</a>
                    </p>

                    <form onSubmit={handleLogin} className="login-form">
                        <div className="input-group-row">
                            <input
                                type="text"
                                placeholder="First name"
                                className="login-input"
                                value={firstName}
                                onChange={(e) => setFirstName(e.target.value)}
                                required
                            />
                            <input
                                type="text"
                                placeholder="Last name"
                                className="login-input"
                                value={lastName}
                                onChange={(e) => setLastName(e.target.value)}
                                required
                            />
                        </div>

                        <input type="email" placeholder="Email" className="login-input full-width" required />

                        <div className="password-wrapper">
                            <input
                                type={showPassword ? 'text' : 'password'}
                                placeholder="Enter your password"
                                className="login-input full-width"
                                required
                            />
                            <button
                                type="button"
                                className="password-toggle"
                                onClick={() => setShowPassword((prev) => !prev)}
                                aria-label={showPassword ? 'Hide password' : 'Show password'}
                            >
                                {showPassword ? (
                                    /* Eye-off icon */
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M3.70711 2.29289C3.31658 1.90237 2.68342 1.90237 2.29289 2.29289C1.90237 2.68342 1.90237 3.31658 2.29289 3.70711L20.2929 21.7071C20.6834 22.0976 21.3166 22.0976 21.7071 21.7071C22.0976 21.3166 22.0976 20.6834 21.7071 20.2929L3.70711 2.29289Z" fill="currentColor" />
                                        <path d="M5.71 6.88L7.17 8.34C5.97 9.29 5 10.55 4.35 12C5.97 15.61 9.22 18 12 18C13.15 18 14.29 17.67 15.31 17.12L17.12 18.93C15.58 19.64 13.87 20 12 20C7.31 20 3.33 17 1.71 13C2.49 11.05 3.84 9.36 5.71 6.88Z" fill="currentColor" />
                                        <path d="M12 6C16.69 6 20.67 9 22.3 13C21.74 14.39 20.93 15.63 19.94 16.65L14.41 11.12C14.78 10.48 15 9.77 15 9C15 7.34 13.66 6 12 6C11.23 6 10.52 6.22 9.88 6.59L7.35 4.06C8.78 3.4 10.34 3 12 6Z" fill="currentColor" />
                                    </svg>
                                ) : (
                                    /* Eye icon */
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M12 5C7.30558 5 3.32839 8.00695 1.70508 12C3.32839 15.993 7.30558 19 12 19C16.6944 19 20.6716 15.993 22.2949 12C20.6716 8.00695 16.6944 5 12 5ZM12 17C9.23858 17 7 14.7614 7 12C7 9.23858 9.23858 7 12 7C14.7614 7 17 9.23858 17 12C17 14.7614 14.7614 17 12 17ZM12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9Z" fill="currentColor" />
                                    </svg>
                                )}
                            </button>
                        </div>

                        <label className="terms-checkbox">
                            <input type="checkbox" required />
                            <span>I agree to the <a href="#" className="login-link">Terms & Conditions</a></span>
                        </label>

                        <button type="submit" className="login-submit-btn">
                            Create account
                        </button>
                    </form>

                    {authError && <p className="login-error">{authError}</p>}

                    <div className="login-divider">
                        <span>Or register with</span>
                    </div>

                    <div className="social-login-group">
                        <button
                            type="button"
                            className="social-btn google-btn"
                            onClick={() => loginWithGoogle()}
                            disabled={isAuthenticating}
                        >
                            {/* Google "G" SVG logo */}
                            <svg className="social-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z" fill="#4285F4" />
                                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                            </svg>
                            {isAuthenticating ? 'Signing in...' : 'Google'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
