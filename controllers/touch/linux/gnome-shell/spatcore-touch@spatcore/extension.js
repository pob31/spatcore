// GNOME Shell's touchscreen gestures (three-finger swipes for workspaces and
// the overview, the top-edge drag that leaves fullscreen, the bottom-edge drag
// for the on-screen keyboard) are gesture actions on the stage, in the capture
// phase: they see every touch first, and once one recognises a gesture the
// fingers are cancelled for the application underneath. A multitouch
// application then loses touches mid-move, and no Wayland or X11 protocol lets
// it opt out. This extension turns the stage gestures off while a window of a
// registered application is open, and gives each its own state back when the
// last one closes.
//
// Applications register by WM_CLASS: one file per application, named after it,
// in ~/.config/spatcore/touch/gnome-gesture-hold/ (gesture-hold.sh next to this
// folder writes and removes them). The folder is watched, so an application
// registered while GNOME runs is held at once.

import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

const REGISTRY = GLib.build_filenamev(
    [GLib.get_user_config_dir(), 'spatcore', 'touch', 'gnome-gesture-hold']);

function isGesture(action) {
    return (Clutter.GestureAction !== undefined && action instanceof Clutter.GestureAction)
        || (Clutter.Gesture !== undefined && action instanceof Clutter.Gesture);
}

function readRegistry() {
    const classes = new Set();
    try {
        const children = Gio.File.new_for_path(REGISTRY).enumerate_children(
            'standard::name,standard::type', Gio.FileQueryInfoFlags.NONE, null);
        for (let info; (info = children.next_file(null)) !== null;) {
            const name = info.get_name();
            if (info.get_file_type() === Gio.FileType.REGULAR && !name.startsWith('.'))
                classes.add(name);
        }
        children.close(null);
    } catch (e) {
        // No folder yet: nothing registered.
    }
    return classes;
}

export default class SpatcoreTouchExtension extends Extension {
    enable() {
        this._windows = new Map(); // Meta.Window -> signal ids
        this._held = new Map();    // gesture action -> {id, wanted}
        GLib.mkdir_with_parents(REGISTRY, 0o755);
        this._classes = readRegistry();
        this._monitor = Gio.File.new_for_path(REGISTRY)
            .monitor_directory(Gio.FileMonitorFlags.NONE, null);
        this._monitorId = this._monitor.connect('changed', () => {
            this._classes = readRegistry();
            this._update();
        });
        this._createdId = global.display.connect('window-created',
            (_display, window) => this._track(window));
        for (const actor of global.get_window_actors())
            this._track(actor.meta_window);
    }

    disable() {
        this._monitor.disconnect(this._monitorId);
        this._monitor.cancel();
        this._monitor = null;
        global.display.disconnect(this._createdId);
        for (const [window, ids] of this._windows)
            ids.forEach(id => window.disconnect(id));
        this._windows.clear();
        this._release();
    }

    _track(window) {
        if (!window || this._windows.has(window))
            return;
        this._windows.set(window, [
            window.connect('notify::wm-class', () => this._update()),
            window.connect('unmanaged', () => {
                this._windows.get(window)?.forEach(id => window.disconnect(id));
                this._windows.delete(window);
                this._update();
            }),
        ]);
        this._update();
    }

    _update() {
        const open = [...this._windows.keys()].some(w =>
            this._classes.has(w.get_wm_class()) || this._classes.has(w.get_wm_class_instance()));
        if (open)
            this._hold();
        else
            this._release();
    }

    _hold() {
        for (const action of global.stage.get_actions()) {
            if (!isGesture(action) || this._held.has(action))
                continue;
            // The shell re-enables a gesture whenever its swipe tracker is
            // re-enabled (bind_property); remember that wish and keep it off.
            const entry = {wanted: action.enabled, id: 0};
            entry.id = action.connect('notify::enabled', () => {
                if (action.enabled) {
                    entry.wanted = true;
                    action.enabled = false;
                }
            });
            action.enabled = false;
            this._held.set(action, entry);
        }
    }

    _release() {
        for (const [action, entry] of this._held) {
            action.disconnect(entry.id);
            action.enabled = entry.wanted;
        }
        this._held.clear();
    }
}
