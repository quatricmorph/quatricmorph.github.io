import type { MetadataRoute } from 'next'
import { readdirSync, statSync, readFileSync } from 'fs'
import path from 'path'

export const dynamic = "force-static"

// Base URL for the site
const BASE_URL = process.env.SITE_URL || 'https://vietllm.pages.dev'

// Function to get file modification time
function getFileModTime(filePath: string): Date {
    try {
        return statSync(filePath).mtime
    } catch {
        return new Date()
    }
}

// Function to recursively find files
function findFiles(dir: string, extension: RegExp, basePath: string = ''): string[] {
    const files: string[] = []
    
    try {
        const items = readdirSync(path.join(process.cwd(), dir), { withFileTypes: true })
        
        for (const item of items) {
            const fullPath = path.join(dir, item.name)
            const relativePath = basePath ? path.join(basePath, item.name) : item.name
            
            if (item.isDirectory()) {
                // Recursively search subdirectories
                files.push(...findFiles(fullPath, extension, relativePath))
            } else if (item.isFile() && extension.test(item.name)) {
                files.push(relativePath)
            }
        }
    } catch (error) {
        console.warn(`Could not read directory ${dir}:`, error)
    }
    
    return files
}

// Function to convert file path to URL path
function filePathToUrl(filePath: string): string {
    // Remove file extensions and handle page files
    let urlPath = filePath
        .replace(/\/page\.(mdx|tsx)$/, '')
        .replace(/\.(mdx|tsx)$/, '')
    
    // Handle root page
    if (urlPath === 'page' || urlPath === '') {
        return ''
    }
    
    return urlPath
}

// Function to determine priority based on path
function getPriority(urlPath: string): number {
    if (urlPath === '') return 1.0 // Homepage
    if (urlPath === 'about') return 0.8
    if (urlPath === 'posts') return 0.7
    if (urlPath.startsWith('posts/')) return 0.6 // Individual posts
    return 0.5
}

// Function to determine change frequency
function getChangeFrequency(urlPath: string): 'always' | 'hourly' | 'daily' | 'weekly' | 'monthly' | 'yearly' | 'never' {
    if (urlPath === '') return 'monthly' // Homepage
    if (urlPath === 'about') return 'yearly'
    if (urlPath === 'posts') return 'weekly'
    if (urlPath.startsWith('posts/')) return 'monthly' // Individual posts
    return 'monthly'
}

// Function to find images in public directory
function findImages(dir: string = 'public'): string[] {
    const imageExtensions = /\.(jpg|jpeg|png|gif|webp|svg)$/i
    const images: string[] = []
    
    try {
        const fullDir = path.join(process.cwd(), dir)
        const items = readdirSync(fullDir, { withFileTypes: true })
        
        for (const item of items) {
            const itemPath = path.join(dir, item.name)
            
            if (item.isDirectory() && !item.name.startsWith('.') && !item.name.startsWith('_')) {
                // Recursively search subdirectories, excluding hidden and system directories
                images.push(...findImages(itemPath))
            } else if (item.isFile() && imageExtensions.test(item.name)) {
                // Convert to URL path (remove 'public' prefix)
                const urlPath = itemPath.replace(/^public\//, '/')
                images.push(urlPath)
            }
        }
    } catch (error) {
        console.warn(`Could not read directory ${dir}:`, error)
    }
    
    return images
}

// Function to extract images from markdown content
function extractImagesFromMdx(filePath: string): string[] {
    const images: string[] = []
    
    try {
        const content = readFileSync(filePath, 'utf-8')
        
        // Match markdown image syntax: ![alt](src)
        const markdownImageRegex = /!\[.*?\]\(([^)]+)\)/g
        // Match HTML img tags: <img src="..." />
        const htmlImageRegex = /<img[^>]+src=["']([^"']+)["'][^>]*>/g
        
        let match
        
        // Extract markdown images
        while ((match = markdownImageRegex.exec(content)) !== null) {
            let imagePath = match[1]
            // Convert relative paths to absolute URLs
            if (!imagePath.startsWith('http') && !imagePath.startsWith('/')) {
                imagePath = `/${imagePath}`
            }
            if (!imagePath.startsWith('http')) {
                images.push(imagePath)
            }
        }
        
        // Extract HTML images
        while ((match = htmlImageRegex.exec(content)) !== null) {
            let imagePath = match[1]
            // Convert relative paths to absolute URLs
            if (!imagePath.startsWith('http') && !imagePath.startsWith('/')) {
                imagePath = `/${imagePath}`
            }
            if (!imagePath.startsWith('http')) {
                images.push(imagePath)
            }
        }
    } catch (error) {
        console.warn(`Could not read file ${filePath}:`, error)
    }
    
    return images
}

// Function to get images for a specific page
function getPageImages(filePath: string, urlPath: string): string[] | undefined {
    const images: string[] = []
    
    // Get images from the page content
    const fullPath = path.join(process.cwd(), 'app', filePath)
    const contentImages = extractImagesFromMdx(fullPath)
    
    // Add content images
    contentImages.forEach(imagePath => {
        images.push(`${BASE_URL}${imagePath}`)
    })
    
    // For homepage, also include some key public images (excluding favicons)
    if (urlPath === '') {
        const publicImages = findImages().filter(img => 
            !img.includes('favicon') && 
            !img.includes('manifest') &&
            !img.includes('_pagefind')
        )
        
        publicImages.forEach(imagePath => {
            images.push(`${BASE_URL}${imagePath}`)
        })
    }
    
    return images.length ? images : undefined
}

export default function sitemap(): MetadataRoute.Sitemap {
    try {
        // Find all page files and MDX files in the app directory
        const appFiles = findFiles('app', /\.(mdx|tsx)$/)
        
        // Filter to get relevant files
        const relevantFiles = appFiles.filter(file => {
            // Include page files
            if (file.endsWith('/page.mdx') || file.endsWith('/page.tsx')) {
                return true
            }
            // Include individual post files (but not the posts/page.mdx)
            if (file.startsWith('posts/') && file.endsWith('.mdx') && !file.endsWith('/page.mdx')) {
                return true
            }
            // Include root page.mdx
            if (file === 'page.mdx') {
                return true
            }
            return false
        })
        
        // Get all images from public directory for the homepage
        const allImages = findImages().filter(img => 
            !img.includes('favicon') && 
            !img.includes('manifest') &&
            !img.includes('_pagefind')
        ).map(img => `${BASE_URL}${img}`)
        
        const sitemapEntries: MetadataRoute.Sitemap = relevantFiles.map(filePath => {
            const urlPath = filePathToUrl(filePath)
            const fullPath = path.join(process.cwd(), 'app', filePath)
            const lastModified = getFileModTime(fullPath)
            const images = getPageImages(filePath, urlPath)
            
            const entry: MetadataRoute.Sitemap[0] = {
                url: `${BASE_URL}${urlPath ? `/${urlPath}` : ''}`,
                lastModified,
                changeFrequency: getChangeFrequency(urlPath),
                priority: getPriority(urlPath),
            }
            
            // Add images if any are found
            if (images && images.length > 0) {
                entry.images = images
            } else if (urlPath === '' && allImages.length > 0) {
                // For homepage, include all available images if no content images found
                entry.images = allImages
            }
            
            return entry
        })
        
        // Sort by priority (highest first)
        return sitemapEntries.sort((a, b) => (b.priority || 0) - (a.priority || 0))
        
    } catch (error) {
        console.error('Error generating sitemap:', error)
        // Fallback to static entries if dynamic generation fails
        const fallbackImages = findImages().filter(img => 
            !img.includes('favicon') && 
            !img.includes('manifest') &&
            !img.includes('_pagefind')
        ).map(img => `${BASE_URL}${img}`)
        
        return [
            {
                url: `${BASE_URL}`,
                lastModified: new Date(),
                changeFrequency: 'monthly',
                priority: 1,
                images: fallbackImages.length > 0 ? fallbackImages : undefined,
            },
            {
                url: `${BASE_URL}/about`,
                lastModified: new Date(),
                changeFrequency: 'yearly',
                priority: 0.8,
            },
            {
                url: `${BASE_URL}/posts`,
                lastModified: new Date(),
                changeFrequency: 'weekly',
                priority: 0.7,
            },
        ]
    }
}