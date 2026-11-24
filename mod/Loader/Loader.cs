using System;
using System.IO;
using System.Reflection;
using BepInEx;
using BepInEx.Configuration;
using BepInEx.Logging;
using Mono.Cecil;
using UnityEngine;

namespace RainBot
{
    /// <summary>What the loader gives the core: the plugin's config and log, a MonoBehaviour for
    /// coroutines, and a way to ask for a reload.</summary>
    public interface IHost
    {
        ConfigFile Config { get; }
        ManualLogSource Log { get; }
        MonoBehaviour Mono { get; }
        /// <summary>Reload the core now, even with an agent connected (the `reload` command).</summary>
        void RequestReload();
    }

    /// <summary>The real mod (RainBotCore.dll). Start/Stop must leave nothing behind: every
    /// hook, socket, coroutine and texture is released in Stop.</summary>
    public interface ICore
    {
        void Start(IHost host);
        void Stop();
        void OnGUI();
        /// <summary>False while an agent is connected: a new build waits for it to leave.</summary>
        bool CanReload { get; }
    }

    /// <summary>
    /// Hot reload. Rain World loads mods once per launch, and a launch takes 1-3 minutes under
    /// Proton (and fails while the session is locked). The core is loaded from
    /// BepInEx/rainbot/RainBotCore.dll as bytes, under a new assembly name each time: Mono cannot
    /// unload an assembly, and returns the old one for the same name. The file is checked once a
    /// second, and a new build is swapped in as soon as no agent is connected (at once on the
    /// `reload` command). The config file is re-read on every load. Old assemblies stay in memory
    /// (~0.5 MB each).
    /// </summary>
    [BepInPlugin("rainbot.bridge", "RainBot Bridge", "0.4.0")]
    public sealed class Loader : BaseUnityPlugin, IHost
    {
        private ICore _core;
        private string _path;
        private DateTime _stamp;
        private float _nextCheck;
        private bool _pending, _force;
        private int _loads;

        ConfigFile IHost.Config => Config;
        ManualLogSource IHost.Log => Logger;
        MonoBehaviour IHost.Mono => this;
        public void RequestReload() { _force = true; }

        public void OnEnable()
        {
            _path = Path.Combine(Path.Combine(Paths.BepInExRootPath, "rainbot"), "RainBotCore.dll");
            Load();
        }

        private void Load()
        {
            try
            {
                _stamp = File.GetLastWriteTimeUtc(_path);
                byte[] raw = File.ReadAllBytes(_path);
                Assembly asm;
                using (MemoryStream src = new MemoryStream(raw))
                using (AssemblyDefinition def = AssemblyDefinition.ReadAssembly(src))
                using (MemoryStream dst = new MemoryStream())
                {
                    def.Name.Name = "RainBotCore-" + DateTime.UtcNow.Ticks;
                    def.Write(dst);
                    asm = Assembly.Load(dst.ToArray());
                }
                Type t = asm.GetType("RainBot.Plugin", true);
                Config.Reload();
                ICore core = (ICore)Activator.CreateInstance(t);
                core.Start(this);
                _core = core;
                _loads++;
                Logger.LogMessage("core loaded (#" + _loads + ", built " + _stamp.ToLocalTime().ToString("HH:mm:ss") + ")");
            }
            catch (Exception e)
            {
                _core = null;
                Logger.LogError("core load failed: " + e);
            }
        }

        private void Unload()
        {
            if (_core == null) return;
            try { _core.Stop(); }
            catch (Exception e) { Logger.LogError("core stop failed: " + e); }
            _core = null;
        }

        public void Update()
        {
            if (_force) { _force = false; _pending = false; Unload(); Load(); return; }
            if (Time.unscaledTime < _nextCheck) return;
            _nextCheck = Time.unscaledTime + 1f;
            DateTime s;
            try { s = File.GetLastWriteTimeUtc(_path); } catch (Exception) { return; }
            if (s != _stamp && !_pending)
            {
                _pending = true;
                Logger.LogMessage("new core build on disk; swapping it in when no agent is connected");
            }
            if (_pending && (_core == null || _core.CanReload))
            {
                _pending = false;
                Unload();
                Load();
            }
        }

        public void OnGUI() { if (_core != null) _core.OnGUI(); }
        public void OnDisable() { Unload(); }
        public void OnApplicationQuit() { Unload(); }
    }
}
