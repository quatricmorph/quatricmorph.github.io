# ViệtLLM Documentation

A modern documentation and blog site built with [Nextra 4](https://nextra.site) - a powerful static site generator framework based on Next.js 15 with App Router.

## 🚀 Features

- **Nextra 4 with App Router**: Latest Nextra framework with Next.js 15 App Router
- **Markdown & MDX Support**: Write content in markdown with React component support  
- **Search Functionality**: Pagefind integration for fast client-side search
- **SEO Friendly**: Automatic meta tags and sitemap generation
- **Fast Performance**: Built on Next.js 15 for optimal performance
- **Easy Content Management**: Simple file-based content management with App Router
- **Syntax Highlighting**: Beautiful code highlighting for multiple languages
- **Math Support**: KaTeX support for mathematical expressions
- **Dark/Light Mode**: Theme switching support
- **Static Export**: Optimized for GitHub Pages deployment

## 📁 Project Structure

```
vietnamllm.github.io/
├── app/
│   ├── layout.tsx       # Root layout with Nextra theme
│   ├── page.mdx         # Homepage
│   ├── globals.css      # Global styles
│   ├── about/
│   │   └── page.mdx     # About page
│   └── posts/
│       ├── page.mdx     # Posts index page
│       └── *.mdx        # Blog posts
├── public/
│   ├── _pagefind/       # Pagefind search index (auto-generated)
│   └── favicon/         # Favicon files
├── scripts/
│   └── new-post.js      # Script to create new blog posts
├── templates/
│   └── blog-post.md     # Template for new posts
├── next.config.mjs      # Next.js configuration
├── mdx-components.tsx   # Global MDX components
└── package.json
```

## 🛠️ Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/VietnamLLM/vietnamllm.github.io.git
cd vietnamllm.github.io
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## ✏️ Creating Content

### Method 1: Using the Script (Recommended)

Create a new blog post interactively:

```bash
npm run new-post
```

The script will ask for:

- Post title
- Description  
- Author name
- Tags

### Method 2: Manual Creation

1. Create a new `.mdx` file in `app/posts/`
2. Add frontmatter with metadata:

```markdown
---
title: Your Post Title
date: 2025/09/27
description: Brief description for SEO
tag: tag1, tag2, tag3
author: Your Name
---

# Your Post Title

Your content here...
```

### Method 3: Using Template

Copy the template file:

```bash
cp templates/blog-post.md app/posts/your-new-post.mdx
```

Then edit the file with your content.

## 📝 Writing Guidelines

### Frontmatter Fields

- `title`: Post title (required)
- `date`: Publication date in YYYY/MM/DD format (required)
- `description`: SEO description (recommended)
- `tag`: Comma-separated tags (optional)
- `author`: Author name (optional)

### Markdown Features

- **Headers**: Use `#`, `##`, `###` for headings
- **Code blocks**: Use triple backticks with language specification
- **Math**: Use `$inline math$` or `$$block math$$`
- **Links**: `[text](url)`
- **Images**: `![alt](url)`
- **Lists**: Use `-` or `1.` for lists
- **Tables**: Standard markdown table syntax

### MDX Components

You can use React components in `.mdx` files:

```mdx
import { Callout } from 'nextra/components'

<Callout type="info">
This is an info callout
</Callout>
```

## 🔍 Search Functionality

This site includes client-side search powered by [Pagefind](https://pagefind.app/):

- **Automatic indexing**: Content is indexed during the build process
- **Fast search**: Client-side search with no server required
- **GitHub Pages compatible**: Works perfectly with static hosting

The search index is automatically generated and stored in `out/_pagefind/` after build.

### Search Setup

The search functionality requires:

1. **Build with indexing**: `npm run build` (includes postbuild script)
2. **Search files**: Available at `/_pagefind/` when deployed
3. **Integration**: Nextra theme handles search UI automatically

### Troubleshooting Search

If search is not working:

- Verify `out/_pagefind/` directory exists after build
- Check that Pagefind files are deployed to `/_pagefind/` on your site
- Ensure content has proper HTML structure (Pagefind looks for `data-pagefind-body`)

## 🗺️ Sitemap

The site automatically generates an XML sitemap for SEO:

### Dynamic Sitemap Route

- **URL**: `/sitemap.xml` 
- **Auto-discovery**: Scans all MDX files in the `app/` directory
- **Metadata**: Extracts frontmatter data (title, date) from posts
- **SEO optimized**: Includes priority, change frequency, and last modified dates

### Sitemap Features

- **Automatic page discovery**: Finds all MDX pages and posts
- **Priority assignment**: 
  - Homepage: `1.0` (highest priority)
  - Blog posts: `0.8` 
  - Other pages: `0.5-0.7`
- **Change frequency**: Based on content type (daily/weekly/monthly)
- **Last modified**: Uses frontmatter date or current date

### Static Sitemap Generation

For static hosting, a sitemap file is also generated during build:

```bash
npm run sitemap  # Generate sitemap.xml in out/ directory
```

The sitemap is accessible at: `https://vietnamllm.github.io/sitemap.xml`

## 🎨 Customization

### Theme Configuration

Edit `theme.config.jsx` to customize:

- Site footer
- Navigation links
- Meta tags
- Dark mode settings

### Styling

The blog uses Nextra's built-in styling. For custom styles:

1. Create CSS modules in a `styles/` directory
2. Import and use in your components
3. Use Tailwind CSS classes (if enabled)

## 🚀 Deployment

### GitHub Pages (Current Setup)

This repository is configured for automatic deployment to GitHub Pages using GitHub Actions:

1. **Automatic deployment**: Push to `main` branch triggers deployment
2. **Build process**: Next.js builds with static export to `out/` directory  
3. **Search indexing**: Pagefind generates search index from build output
4. **Deploy**: GitHub Actions deploys the static files to GitHub Pages

The site is available at: <https://vietnamllm.github.io>

### Manual Deployment

To build and deploy manually:

```bash
npm run build
```

The static files will be generated in the `out/` directory.

### Alternative Platforms

#### Vercel

1. Connect your GitHub repository to Vercel
2. Vercel will automatically deploy on every push to main

#### Netlify

1. Connect your repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `out`

## 📚 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production with static export
- `npm start` - Start production server  
- `npm run lint` - Run ESLint
- `npm run new-post` - Create new blog post interactively
- `npm run sitemap` - Generate sitemap.xml file (standalone)
- `postbuild` - Generate Pagefind search index and sitemap (runs automatically after build)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-post`
3. Make your changes
4. Commit: `git commit -am 'Add new post'`
5. Push: `git push origin feature/new-post`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Nextra Documentation](https://nextra.site)
- [Next.js Documentation](https://nextjs.org/docs)
- [Markdown Guide](https://www.markdownguide.org/)
- [VietnamLLM GitHub](https://github.com/VietnamLLM)

---

Built with ❤️ by the VietnamLLM community
vietnamllm.github.io
