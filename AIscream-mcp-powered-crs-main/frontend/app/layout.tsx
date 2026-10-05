import './globals.css';
import type { ReactNode } from 'react';

export const metadata = {
  title: 'MCP-Powered CRS',
  description: 'Conversational entertainment recommendations powered by MCP.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
