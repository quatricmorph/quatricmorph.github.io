#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const readline = require('readline')

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

function question(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, resolve)
  })
}

async function createBlogPost() {
  console.log('🚀 Creating a new blog post...\n')
  
  const title = await question('Post title: ')
  const description = await question('Description: ')
  const author = await question('Author (default: Quatricmorph Team): ') || 'Quatricmorph Team'
  const tags = await question('Tags (comma-separated): ')
  
  // Generate filename from title
  const filename = title
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, '-')
  
  // Get current date
  const date = new Date()
  const dateStr = `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')}`
  
  // Create blog post content
  const content = `---
title: ${title}
date: ${dateStr}
description: ${description}
tag: ${tags}
author: ${author}
---

# ${title}

Write your blog post content here...

## Introduction

Start with an engaging introduction.

## Main Content

Add your main content here.

## Conclusion

Wrap up your thoughts.
`

  // Write the file
  const postsDir = path.join(__dirname, '..', 'pages', 'posts')
  const filePath = path.join(postsDir, `${filename}.md`)
  
  fs.writeFileSync(filePath, content)
  
  console.log(`\n✅ Blog post created successfully!`)
  console.log(`📝 File: ${filePath}`)
  console.log(`🌐 URL: /posts/${filename}`)
  console.log(`\nYou can now edit the file and add your content.`)
  
  rl.close()
}

if (require.main === module) {
  createBlogPost().catch(console.error)
}

module.exports = { createBlogPost }