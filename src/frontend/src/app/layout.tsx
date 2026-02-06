import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "EagleEye – Inventory Dashboard",
  description: "Inventory management insights and recommendations",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`min-h-screen bg-slate-950 text-slate-50 ${outfit.className}`} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}