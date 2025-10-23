import { useMDXComponents as getThemeComponents } from 'nextra-theme-docs' // nextra-theme-blog or your custom theme
import { MDXComponents } from 'mdx/types'


const themeComponents = getThemeComponents()

// Get the default MDX components

export function useMDXComponents(components: MDXComponents): MDXComponents {
    return {
        ...themeComponents,
        ...components,
    }
}