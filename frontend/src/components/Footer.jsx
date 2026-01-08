import React from 'react';

export default function Footer() {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="main-footer">
            <div className="footer-bottom">
                <p>
                    &copy; {currentYear} <strong>Talkify AI</strong>. All rights reserved.
                    <span>Multimodal AI for Empowerment and Accessibility.</span>
                </p>
            </div>
        </footer>
    );
}
