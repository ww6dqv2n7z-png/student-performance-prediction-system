import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'MTU CEIT Student Academic Support System',
  description: 'A secure local ANN decision-support system for the MTU CEIT department.',
  openGraph: {
    title: 'MTU CEIT Student Academic Support System',
    description: 'A six-year departmental pilot for early academic support.',
    images: ['http://localhost:3000/og.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'MTU CEIT Student Academic Support System',
    description: 'A six-year departmental pilot for early academic support.',
    images: ['http://localhost:3000/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
