import { ReactNode } from 'react'

export default function PostsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="nx-max-w-4xl nx-mx-auto nx-px-6">
      <nav className="nx-flex nx-items-center nx-justify-between nx-py-4 nx-border-b nx-border-gray-200">
        <div className="nx-flex nx-items-center nx-space-x-4">
          <a href="/" className="nx-text-xl nx-font-bold">Quatricmorph</a>
        </div>
        <div className="nx-flex nx-space-x-6">
          <a href="/" className="nx-text-gray-600 hover:nx-text-gray-900">Home</a>
          <a href="/posts" className="nx-text-blue-600 hover:nx-text-blue-800">Posts</a>
          <a href="/docs" className="nx-text-gray-600 hover:nx-text-gray-900">Docs</a>
          <a href="/about" className="nx-text-gray-600 hover:nx-text-gray-900">About</a>
        </div>
      </nav>
      <main className="nx-py-8">
        {children}
      </main>
    </div>
  )
}