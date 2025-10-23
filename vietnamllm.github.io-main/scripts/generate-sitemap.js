#!/usr/bin/env node

import { readdir, readFile, writeFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

const SITE_URL = 'https://vietnamllm.github.io';
const OUTPUT_DIR = 'out';

// Function to extract frontmatter from MDX files
function extractFrontmatter(content) {
  const frontmatterRegex = /^---\n([\s\S]*?)\n---/;
  const match = content.match(frontmatterRegex);
  
  if (!match) return {};
  
  const frontmatter = {};
  const lines = match[1].split('\n');
  
  for (const line of lines) {
    const [key, ...valueParts] = line.split(':');
    if (key && valueParts.length > 0) {
      const value = valueParts.join(':').trim();
      frontmatter[key.trim()] = value;
    }
  }
  
  return frontmatter;
}

// Function to get all MDX files recursively
async function getMDXFiles(dir, baseDir = dir) {
  const files = [];
  
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = join(dir, entry.name);
      const relativePath = fullPath.replace(baseDir + '/', '');
      
      if (entry.isDirectory()) {
        const subFiles = await getMDXFiles(fullPath, baseDir);
        files.push(...subFiles);
      } else if (entry.name.endsWith('.mdx')) {
        files.push({
          path: fullPath,
          relativePath: relativePath
        });
      }
    }
  } catch (error) {
    console.warn(`Warning: Could not read directory ${dir}:`, error.message);
  }
  
  return files;
}

// Function to convert file path to URL path
function filePathToUrlPath(filePath) {
  // Remove .mdx extension
  let urlPath = filePath.replace(/\.mdx$/, '');
  
  // Handle page.mdx files (convert to directory index)
  urlPath = urlPath.replace(/\/page$/, '');
  
  // Handle root page
  if (urlPath === 'page' || urlPath === '') {
    return '/';
  }
  
  // Ensure leading slash
  if (!urlPath.startsWith('/')) {
    urlPath = '/' + urlPath;
  }
  
  // Ensure trailing slash for consistency
  if (urlPath !== '/' && !urlPath.endsWith('/')) {
    urlPath += '/';
  }
  
  return urlPath;
}

// Function to generate sitemap XML
function generateSitemapXML(pages) {
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map(page => `  <url>
    <loc>${SITE_URL}${page.url}</loc>
    <lastmod>${page.lastmod}</lastmod>
    <changefreq>${page.changefreq}</changefreq>
    <priority>${page.priority}</priority>
  </url>`).join('\n')}
</urlset>`;

  return sitemap;
}

// Main function
async function generateSitemap() {
  console.log('🗺️  Generating sitemap...');
  
  const appDir = join(projectRoot, 'app');
  const outputPath = join(projectRoot, OUTPUT_DIR, 'sitemap.xml');
  
  if (!existsSync(appDir)) {
    console.error('❌ App directory not found:', appDir);
    process.exit(1);
  }
  
  // Get all MDX files
  const mdxFiles = await getMDXFiles(appDir);
  console.log(`📄 Found ${mdxFiles.length} MDX files`);
  
  const pages = [];
  const now = new Date().toISOString().split('T')[0]; // YYYY-MM-DD format
  
  for (const file of mdxFiles) {
    try {
      const content = await readFile(file.path, 'utf-8');
      const frontmatter = extractFrontmatter(content);
      
      // Convert file path to URL path
      const relativePath = file.relativePath.replace('app/', '');
      const urlPath = filePathToUrlPath(relativePath);
      
      // Determine page priority and change frequency
      let priority = '0.5';
      let changefreq = 'weekly';
      
      if (urlPath === '/') {
        priority = '1.0';
        changefreq = 'daily';
      } else if (urlPath.startsWith('/posts/') && urlPath !== '/posts/') {
        priority = '0.8';
        changefreq = 'monthly';
      } else if (urlPath === '/posts/' || urlPath === '/about/') {
        priority = '0.7';
        changefreq = 'weekly';
      }
      
      // Use frontmatter date if available, otherwise use current date
      let lastmod = now;
      if (frontmatter.date) {
        try {
          const date = new Date(frontmatter.date.replace(/\//g, '-'));
          if (!isNaN(date.getTime())) {
            lastmod = date.toISOString().split('T')[0];
          }
        } catch (e) {
          console.warn(`Warning: Invalid date format in ${file.relativePath}:`, frontmatter.date);
        }
      }
      
      pages.push({
        url: urlPath,
        lastmod,
        changefreq,
        priority,
        title: frontmatter.title || 'Untitled'
      });
      
      console.log(`  ✅ ${urlPath} (${frontmatter.title || 'Untitled'})`);
      
    } catch (error) {
      console.warn(`Warning: Could not process ${file.relativePath}:`, error.message);
    }
  }
  
  // Sort pages by URL for consistency
  pages.sort((a, b) => a.url.localeCompare(b.url));
  
  // Generate sitemap XML
  const sitemapXML = generateSitemapXML(pages);
  
  // Ensure output directory exists
  const outputDir = dirname(outputPath);
  if (!existsSync(outputDir)) {
    console.error('❌ Output directory not found:', outputDir);
    console.log('Make sure to run this script after building the site.');
    process.exit(1);
  }
  
  // Write sitemap
  await writeFile(outputPath, sitemapXML, 'utf-8');
  
  console.log(`✅ Sitemap generated successfully: ${outputPath}`);
  console.log(`📊 Total pages: ${pages.length}`);
  console.log(`🌐 Site URL: ${SITE_URL}`);
}

// Run the script
generateSitemap().catch(error => {
  console.error('❌ Error generating sitemap:', error);
  process.exit(1);
});