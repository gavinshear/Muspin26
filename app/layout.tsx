import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MuSpin - Radial Music Sequencer",
  description: "MuSpin is a radial step sequencer for making music.",
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