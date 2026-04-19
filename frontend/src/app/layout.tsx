import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { AppLayout } from "@/components/layout/app-layout";

export const metadata: Metadata = {
  title: "MediPredict — AI Disease Prediction",
  description:
    "ML-powered disease prediction system using XGBoost, Random Forest, and ResNet-50 for medical diagnosis.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body>
        <Providers>
          <AppLayout>{children}</AppLayout>
        </Providers>
      </body>
    </html>
  );
}
