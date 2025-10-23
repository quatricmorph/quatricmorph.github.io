import React from 'react'
import './globals.css'

export default function HomePageLayout() {
  return (
    <div style={{ 
      maxWidth: '1200px', 
      margin: '0 auto', 
      padding: '2rem',
      minHeight: '100vh'
    }}>
      <nav style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '3rem',
        paddingBottom: '1rem',
        borderBottom: '1px solid #e1e5e9'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <img
            src="/favicon/favicon.svg"
            alt="Quatricmorph"
            style={{ width: '32px', height: '32px' }}
          />
          <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 'bold' }}>Quatricmorph</h1>
        </div>
        <div style={{ display: 'flex', gap: '2rem' }}>
          <a href="/posts" style={{ textDecoration: 'none', color: '#0070f3', fontWeight: '500' }}>
            Posts
          </a>
          <a href="/docs" style={{ textDecoration: 'none', color: '#0070f3', fontWeight: '500' }}>
            Documentation
          </a>
          <a href="/about" style={{ textDecoration: 'none', color: '#0070f3', fontWeight: '500' }}>
            About
          </a>
        </div>
      </nav>
    </div>
  )
}