import React from 'react'
import 'nextra-theme-docs/style.css'

export default function DocsLayout({ 
  children 
}: { 
  children: React.ReactNode 
}) {
    return (
        <div style={{ minHeight: '100vh' }}>
            <nav style={{ 
                backgroundColor: '#fff',
                borderBottom: '1px solid #e1e5e9',
                padding: '1rem 0',
                position: 'sticky',
                top: 0,
                zIndex: 1000
            }}>
                <div style={{ 
                    maxWidth: '1200px', 
                    margin: '0 auto',
                    padding: '0 2rem',
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <img
                            src="/favicon/favicon.svg"
                            alt="ViệtLLM"
                            style={{ width: '24px', height: '24px' }}
                        />
                        <a href="/" style={{ textDecoration: 'none', color: '#000', fontWeight: 'bold', fontSize: '1.25rem' }}>
                            ViệtLLM
                        </a>
                    </div>
                    <div style={{ display: 'flex', gap: '2rem' }}>
                        <a href="/" style={{ textDecoration: 'none', color: '#666', fontWeight: '500' }}>
                            Home
                        </a>
                        <a href="/posts" style={{ textDecoration: 'none', color: '#666', fontWeight: '500' }}>
                            Posts
                        </a>
                        <a href="/docs" style={{ textDecoration: 'none', color: '#0070f3', fontWeight: '500' }}>
                            Documentation
                        </a>
                        <a href="/about" style={{ textDecoration: 'none', color: '#666', fontWeight: '500' }}>
                            About
                        </a>
                    </div>
                </div>
            </nav>
            
            <div style={{ display: 'flex', maxWidth: '1200px', margin: '0 auto' }}>
                <aside style={{ 
                    width: '250px', 
                    padding: '2rem 1rem',
                    borderRight: '1px solid #e1e5e9',
                    minHeight: 'calc(100vh - 80px)'
                }}>
                    <h3 style={{ marginBottom: '1rem', fontSize: '1rem', fontWeight: 'bold' }}>Documentation</h3>
                    <nav>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                            <li style={{ marginBottom: '0.5rem' }}>
                                <a href="/docs" style={{ textDecoration: 'none', color: '#666', fontSize: '0.875rem' }}>
                                    Introduction
                                </a>
                            </li>
                            <li style={{ marginBottom: '0.5rem' }}>
                                <a href="/docs/getting-started" style={{ textDecoration: 'none', color: '#666', fontSize: '0.875rem' }}>
                                    Getting Started
                                </a>
                            </li>
                            <li style={{ marginBottom: '0.5rem' }}>
                                <a href="/docs/api" style={{ textDecoration: 'none', color: '#666', fontSize: '0.875rem' }}>
                                    API Reference
                                </a>
                            </li>
                        </ul>
                    </nav>
                </aside>
                
                <main style={{ flex: 1, padding: '2rem' }}>
                    {children}
                </main>
            </div>
        </div>
    )
}