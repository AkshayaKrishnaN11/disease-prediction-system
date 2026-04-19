"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/lib/store";
import {
  LayoutDashboard,
  Stethoscope,
  ScanLine,
  BarChart3,
  History,
  ChevronLeft,
  Activity,
} from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/predict", label: "Disease Prediction", icon: Stethoscope },
  { href: "/xray", label: "X-Ray Analysis", icon: ScanLine },
  { href: "/metrics", label: "Model Metrics", icon: BarChart3 },
  { href: "/history", label: "History", icon: History },
];

export function Sidebar() {
  const { sidebarOpen, toggleSidebar } = useAppStore();
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen transition-all duration-300 ease-in-out",
        "glass border-r border-border/50",
        sidebarOpen ? "w-64" : "w-20"
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between px-4 border-b border-border/50">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 glow">
            <Activity className="h-5 w-5 text-primary" />
          </div>
          {sidebarOpen && (
            <div className="animate-fade-in">
              <h1 className="text-sm font-bold gradient-text">MediPredict</h1>
              <p className="text-[10px] text-muted-foreground">AI Diagnostics</p>
            </div>
          )}
        </div>
        <button
          onClick={toggleSidebar}
          className="flex h-8 w-8 items-center justify-center rounded-lg hover:bg-secondary transition-colors"
        >
          <ChevronLeft
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform duration-300",
              !sidebarOpen && "rotate-180"
            )}
          />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1 p-3 mt-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-primary/10 text-primary glow"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              )}
            >
              <item.icon className={cn("h-5 w-5 flex-shrink-0", isActive && "text-primary")} />
              {sidebarOpen && (
                <span className="animate-fade-in truncate">{item.label}</span>
              )}
              {isActive && sidebarOpen && (
                <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      {sidebarOpen && (
        <div className="absolute bottom-4 left-3 right-3 animate-fade-in">
          <div className="rounded-xl bg-gradient-to-br from-primary/10 to-accent/10 p-3 border border-primary/20">
            <p className="text-xs font-medium text-foreground">AI-Powered</p>
            <p className="text-[10px] text-muted-foreground mt-0.5">
              XGBoost · RF · ResNet-50
            </p>
          </div>
        </div>
      )}
    </aside>
  );
}
