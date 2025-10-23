# Quick Start Guide

## 🚀 Get Your Blog Running in 5 Minutes

### 1. Start Development Server
```bash
npm run dev
```
Visit: http://localhost:3000

### 2. Create Your First Post
```bash
npm run new-post
```

### 3. Edit Content
- Blog posts: `pages/posts/`
- Pages: `pages/`
- Configuration: `theme.config.jsx`

### 4. Deploy
- **GitHub Pages**: Push to main branch (auto-deploy configured)
- **Vercel**: Connect repository for instant deployment

## 📝 Writing Tips

### Blog Post Structure
```markdown
---
title: Your Amazing Post
date: 2025/09/27
description: SEO-friendly description
tag: topic1, topic2
author: Your Name
---

# Your Amazing Post

Great content starts here...
```

### Using Components
```mdx
import { Callout } from 'nextra/components'

<Callout type="info">
Highlight important information
</Callout>
```

### Code Blocks
````markdown
```javascript
console.log('Hello, Vietnam!')
```
````

### Math Equations
```markdown
Inline: $E = mc^2$

Block:
$$
\sum_{i=1}^n x_i = x_1 + x_2 + \cdots + x_n
$$
```

## 🎨 Customization

- **Theme**: Edit `theme.config.jsx`
- **Styling**: Use Tailwind CSS classes
- **Components**: Create in `components/` directory

## 🔧 Useful Commands

- `npm run dev` - Development server
- `npm run build` - Build for production  
- `npm run new-post` - Create new blog post
- `npm run lint` - Check code quality

## 🆘 Need Help?

- Check the [full README](README.md)
- Visit [Nextra documentation](https://nextra.site)
- Open an issue on GitHub

Happy blogging! 🎉