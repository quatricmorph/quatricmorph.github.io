import { NextRequest } from 'next/server'
import sitemap from '../sitemap'

export async function GET(request: NextRequest) {
  const sitemapData = sitemap()
  
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemapData.map(item => `<url>
<loc>${item.url}</loc>
<lastmod>${item.lastModified ? new Date(item.lastModified).toISOString() : new Date().toISOString()}</lastmod>
<changefreq>${item.changeFrequency || 'monthly'}</changefreq>
<priority>${item.priority || 0.5}</priority>
${item.images ? item.images.map(img => `<image:image xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
<image:loc>${img}</image:loc>
</image:image>`).join('') : ''}
</url>`).join('')}
</urlset>`

  return new Response(xml, {
    headers: {
      'Content-Type': 'application/xml',
    },
  })
}