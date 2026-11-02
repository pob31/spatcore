#!/usr/bin/env bash
# GNOME touchscreen-gesture hold for spatcore applications, per desktop user.
#
#   gesture-hold.sh register <WM_CLASS>     install the extension if needed and
#                                           hold the gestures while that app is open
#   gesture-hold.sh unregister <WM_CLASS>   drop that app; the last one out
#                                           removes the extension
#   gesture-hold.sh status
#
# GNOME Shell's touchscreen gestures take the fingers away from a multitouch
# application (see spatcore-touch@spatcore/extension.js). The extension serves
# every registered spatcore application, so it is installed once and only
# removed when no application is registered any more. Run it as the desktop
# user, not root: the extension, its registry and the GNOME settings are per
# user. Without GNOME Shell it does nothing and exits 0.
set -euo pipefail

UUID="spatcore-touch@spatcore"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_SRC="$HERE/$UUID"
EXT_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/gnome-shell/extensions/$UUID"
REGISTRY="${XDG_CONFIG_HOME:-$HOME/.config}/spatcore/touch/gnome-gesture-hold"

usage() { sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }

# gsettings without the session bus silently writes to a memory backend, so
# only touch GNOME settings when the bus is there (found from ssh too).
session_bus() {
    if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
        local sock="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus"
        [[ -S "$sock" ]] || return 1
        export DBUS_SESSION_BUS_ADDRESS="unix:path=$sock"
    fi
    command -v gsettings >/dev/null 2>&1
}

enabled_add() {
    local cur
    cur=$(gsettings get org.gnome.shell enabled-extensions)
    case "$cur" in
        *"'$UUID'"*) ;;
        "@as []"|"[]") gsettings set org.gnome.shell enabled-extensions "['$UUID']" ;;
        *) gsettings set org.gnome.shell enabled-extensions "${cur%]}, '$UUID']" ;;
    esac
}

enabled_remove() {
    local cur new
    cur=$(gsettings get org.gnome.shell enabled-extensions)
    new=$(printf '%s' "$cur" | sed -e "s/, '$UUID'//" -e "s/'$UUID', //" -e "s/'$UUID'//")
    if [[ "$new" == "[]" ]]; then new="@as []"; fi
    if [[ "$new" != "$cur" ]]; then gsettings set org.gnome.shell enabled-extensions "$new"; fi
}

active() { LC_ALL=C gnome-extensions info "$UUID" 2>/dev/null | grep -q 'State: ACTIVE'; }

registered() { find "$REGISTRY" -maxdepth 1 -type f ! -name '.*' -printf '%f\n' 2>/dev/null | sort; }

cmd="${1:-}"
case "$cmd" in
    register|unregister) [[ $# -eq 2 && -n "$2" && "$2" != */* ]] || usage ;;
    status) ;;
    *) usage ;;
esac
if [[ "$(id -u)" == 0 ]]; then
    echo "gesture-hold: run this as the desktop user, not root." >&2
    exit 1
fi
if ! command -v gnome-shell >/dev/null 2>&1; then
    echo "gesture-hold: GNOME Shell not found; nothing to do."
    exit 0
fi

case "$cmd" in
    register)
        mkdir -p "$EXT_DIR" "$REGISTRY"
        cp "$EXT_SRC/metadata.json" "$EXT_SRC/extension.js" "$EXT_DIR/"
        : > "$REGISTRY/$2"
        if session_bus; then
            # Extensions installed by the user are blocked while this is true.
            gsettings set org.gnome.shell disable-user-extensions false
            enabled_add
            if active; then
                echo "GNOME touch gestures are held while $2 is open, starting now."
            else
                echo "GNOME touch gestures will be held while $2 is open. Log out and back in once for GNOME to load the extension."
            fi
        else
            echo "Extension copied to $EXT_DIR and $2 registered, but GNOME settings are not reachable from here."
            echo "From your desktop session run: $0 register $2   (then log out and back in once)"
        fi
        ;;
    unregister)
        rm -f "$REGISTRY/$2"
        if [[ -z "$(registered)" ]]; then
            rm -rf "$EXT_DIR"
            if session_bus; then enabled_remove; fi
            echo "No spatcore application left registered: GNOME touch-gesture extension removed."
        else
            echo "$2 unregistered; the extension stays for: $(registered | paste -sd ' ' -)"
        fi
        ;;
    status)
        echo "extension: $([[ -d "$EXT_DIR" ]] && echo "installed at $EXT_DIR" || echo "not installed")"
        if session_bus; then echo "state:     $(active && echo active || echo "not active (log out and back in after installing)")"; fi
        echo "registered: $(registered | paste -sd ' ' -)"
        ;;
esac
