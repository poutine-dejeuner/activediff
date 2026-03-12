#!/bin/bash
# Menu interactif GPU GCloud : sélection multi-config + race parallèle.
# La première VM créée gagne, les autres tentatives sont annulées.

set -euo pipefail

REGION="northamerica-northeast1"
ZONES=("a" "b" "c")

VM_NAME="ml-research-montreal"
IMAGE_NAME="ubuntu-accelerator-2404-amd64-with-nvidia-580-v20260225"
IMAGE_PROJECT="ubuntu-os-accelerator-images"
BOOT_DISK_SIZE="100GB"

# ── Catalogue : "GPU|Machine|$/h (total VM+GPU)" ──
CATALOG=(
    "nvidia-tesla-t4|n1-standard-4|0.54"
    "nvidia-l4|g2-standard-4|0.71"
)

BOLD='\033[1m'
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
DIM='\033[2m'
RESET='\033[0m'

# ── 1. Construire la liste de toutes les combinaisons zone × config ──
declare -a ENTRIES=()  # "zone|gpu|machine|price"

for zone_suffix in "${ZONES[@]}"; do
    zone="${REGION}-${zone_suffix}"
    for item in "${CATALOG[@]}"; do
        IFS='|' read -r gpu machine price <<< "$item"
        ENTRIES+=("${zone}|${gpu}|${machine}|${price}")
    done
done

# ── 2. Afficher le menu ──
echo ""
echo -e "${BLUE}${BOLD}  Configurations GPU disponibles – ${REGION}${RESET}"
echo ""
printf "  ${BOLD}%-4s %-30s %-20s %-18s %s${RESET}\n" "#" "ZONE" "GPU" "MACHINE" "\$/h"
echo "  ──── ────────────────────────────── ──────────────────── ────────────────── ─────"

for i in "${!ENTRIES[@]}"; do
    IFS='|' read -r zone gpu machine price <<< "${ENTRIES[$i]}"
    printf "  ${BOLD}%-4s${RESET} %-30s %-20s %-18s ${GREEN}%s${RESET}\n" \
        "$((i+1))" "$zone" "$gpu" "$machine" "\$${price}"
done

echo ""
echo -e "  ${DIM}Sélectionner une ou plusieurs configs (ex: ${BOLD}1,3,5${RESET}${DIM} ou ${BOLD}all${RESET}${DIM})${RESET}"
echo -e "  ${DIM}Les configs choisies seront tentées en parallèle ; la 1ère qui réussit gagne.${RESET}"
echo ""

while true; do
    read -rp "  Choix: " input
    input=$(echo "$input" | tr -d ' ')

    if [[ "$input" == "q" ]]; then
        echo "  Annulé."
        exit 0
    fi

    # Expand "all"
    if [[ "$input" == "all" ]]; then
        SELECTED=()
        for i in "${!ENTRIES[@]}"; do SELECTED+=("$i"); done
        break
    fi

    # Parse comma-separated numbers
    SELECTED=()
    valid=true
    IFS=',' read -ra parts <<< "$input"
    for p in "${parts[@]}"; do
        if [[ "$p" =~ ^[0-9]+$ ]] && [ "$p" -ge 1 ] && [ "$p" -le ${#ENTRIES[@]} ]; then
            SELECTED+=("$((p-1))")
        else
            echo -e "  ${RED}Invalide: $p${RESET}"
            valid=false
            break
        fi
    done
    $valid && [ ${#SELECTED[@]} -gt 0 ] && break
done

echo ""
echo -e "${BLUE}${BOLD}  Lancement de ${#SELECTED[@]} tentative(s) en parallèle...${RESET}"
echo ""

# ── 3. Race parallèle ──
LOCKDIR=$(mktemp -d)
WINNER_FILE="${LOCKDIR}/winner"
PIDS=()

try_create() {
    local idx="$1"
    IFS='|' read -r zone gpu machine price <<< "${ENTRIES[$idx]}"

    # Si un autre process a déjà gagné, on arrête
    [ -f "$WINNER_FILE" ] && return 1

    echo -e "  ${YELLOW}[TRY]${RESET} $machine + $gpu  zone=$zone  (\$${price}/h)"

    output=$(gcloud compute instances create "$VM_NAME" \
        --zone="$zone" \
        --machine-type="$machine" \
        --accelerator="type=${gpu},count=1" \
        --image="$IMAGE_NAME" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --maintenance-policy=TERMINATE \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --quiet 2>&1) && {
            # Écriture atomique du gagnant
            echo "${zone}|${gpu}|${machine}|${price}" > "$WINNER_FILE"
            echo -e "  ${GREEN}${BOLD}[WIN]${RESET} $machine + $gpu  zone=$zone"
            return 0
        } || {
            [ -f "$WINNER_FILE" ] && return 1
            echo -e "  ${RED}[FAIL]${RESET} $machine + $gpu  zone=$zone"
            return 1
        }
}

for idx in "${SELECTED[@]}"; do
    try_create "$idx" &
    PIDS+=($!)
done

# Attendre qu'un process gagne ou que tous échouent
while true; do
    # Vérifier s'il y a un gagnant
    if [ -f "$WINNER_FILE" ]; then
        # Tuer les processus restants
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait "${PIDS[@]}" 2>/dev/null || true
        break
    fi

    # Vérifier si tous les processus sont terminés
    all_done=true
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            all_done=false
            break
        fi
    done
    $all_done && break

    sleep 0.5
done

echo ""

if [ -f "$WINNER_FILE" ]; then
    IFS='|' read -r zone gpu machine price < "$WINNER_FILE"
    echo -e "${GREEN}${BOLD}  ✓ VM '${VM_NAME}' créée !${RESET}"
    echo -e "    Zone:    ${zone}"
    echo -e "    Machine: ${machine}"
    echo -e "    GPU:     ${gpu}"
    echo -e "    Coût:    \$${price}/h"
    echo ""
    echo -e "  ${YELLOW}gcloud compute ssh ${VM_NAME} --zone=${zone}${RESET}"
else
    echo -e "${RED}${BOLD}  ✗ Aucune VM n'a pu être créée.${RESET}"
    echo -e "  ${DIM}Toutes les zones/configs sélectionnées ont échoué (stock épuisé ou quota).${RESET}"
    exit 1
fi

rm -rf "$LOCKDIR"
