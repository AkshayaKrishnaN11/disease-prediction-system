"use client";

import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Sidebar } from "@/components/layout/sidebar";

export function AppLayout({ children }: { children: React.ReactNode }) {
  const { sidebarOpen } = useAppStore();

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main
        className={cn(
          "min-h-screen transition-all duration-300 ease-in-out",
          sidebarOpen ? "ml-64" : "ml-20"
        )}
      >
        <div className="p-6 lg:p-8">{children}</div>
      </main>
    </div>
  );
}
