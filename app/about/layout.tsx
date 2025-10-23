import React from 'react'

export default function AboutLayout({
  children,
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
            <a href="/docs" style={{ textDecoration: 'none', color: '#666', fontWeight: '500' }}>
              Documentation
            </a>
            <a href="/about" style={{ textDecoration: 'none', color: '#0070f3', fontWeight: '500' }}>
              About
            </a>
          </div>
        </div>
      </nav>
      
      <main style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
        {children}
      </main>
    </div>
  )
}