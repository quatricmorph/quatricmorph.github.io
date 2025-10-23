import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Banner, Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import './globals.css'
import React from 'react'

export const metadata = {
    title: 'ViệtLLM',
    description: 'ViệtLLM Documentation'
}

const banner = <Banner storageKey="vietnamllm-banner">Welcome to ViệtLLM 🎉</Banner>

const navbar = (
    <Navbar
        logo={
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <img
                    src="/favicon/favicon.svg"
                    alt="ViệtLLM"
                    style={{ width: '24px', height: '24px' }}
                />
                <b>ViệtLLM</b>
            </div>
        }
    // ... Your additional navbar options
    />
)

const footer = <Footer>MIT {new Date().getFullYear()} © ViệtLLM.</Footer>

export default async function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html
            // Not required, but good for SEO
            lang="en"
            // Required to be set
            dir="ltr"
            // Suggested by `next-themes` package https://github.com/pacocoursey/next-themes#with-app
            suppressHydrationWarning
        >
            <Head >
                <link rel="icon" type="image/png" href="/favicon/favicon-96x96.png" sizes="96x96" />
                <link rel="icon" type="image/svg+xml" href="/favicon/favicon.svg" />
                <link rel="shortcut icon" href="/favicon/favicon.ico" />
                <link rel="apple-touch-icon" sizes="180x180" href="/favicon/apple-touch-icon.png" />
                <link rel="manifest" href="/favicon/site.webmanifest" />
                {/* Preload Fira Code font for better performance */}
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
                <link rel="preload" href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap" as="style" />
                {/* Your additional tags should be passed as `children` of `<Head>` element */}
            </Head>
            <body>
                <Layout
                    banner={banner}
                    navbar={navbar}
                    pageMap={await getPageMap()}
                    docsRepositoryBase="https://github.com/VietnamLLM/vietnamllm.github.io/tree/main"
                    footer={footer}
                // ... Your additional layout options
                >
                    {children}
                </Layout>
            </body>
        </html>
    )
}