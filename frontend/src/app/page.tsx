import { Sidebar } from "@/components/Sidebar";
import { NeuralConsole } from "@/components/NeuralConsole";
import { InteractionZone } from "@/components/InteractionZone";

export default function Home() {
  return (
    <div className="flex h-screen w-full bg-black overflow-hidden font-sans selection:bg-primary/30 selection:text-primary">
      {/* Column A: Left Sidebar */}
      <Sidebar />

      {/* Column B: Center Stage */}
      <InteractionZone />

      {/* Column C: Right Sidebar */}
      <NeuralConsole />
    </div>
  );
}
