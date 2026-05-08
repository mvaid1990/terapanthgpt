// src/app/layout.tsx
// Root layout. Sets dark background and initialises session on first load.

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Terapanth Learning Platform",
  description: "Guided learning for Jain philosophy",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
