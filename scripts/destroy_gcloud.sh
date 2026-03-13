#!/bin/bash
# Liste les VMs actives et permet de les détruire via un menu interactif.

set -euo pipefail

BOLD='\033[1m'
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
DIM='\033[2m'
RESET='\033[0m'

echo -e "\n${BLUE}${BOLD}  Instances VM actives${RESET}\n"

# Récupérer les VMs au format TSV
RAW=$(gcloud compute instances list \
    --format="value(name,zone,machineType.basename(),status,networkInterfaces[0].accessConfigs[0].natIP)" \
    2>/dev/null)

if [ -z "$RAW" ]; then
    echo -e "  ${DIM}Aucune instance trouvée.${RESET}"
    exit 0
fi

declare -a NAMES=()
declare -a ZONES=()
declare -a LINES=()

while IFS=$'\t' read -r name zone machine status ip; do
    NAMES+=("$name")
    ZONES+=("$zone")

    if [[ "$status" == "RUNNING" ]]; then
        st="${GREEN}${status}${RESET}"
    elif [[ "$status" == "TERMINATED" || "$status" == "STOPPED" ]]; then
        st="${RED}${status}${RESET}"
    else
        st="${YELLOW}${status}${RESET}"
    fi

    LINES+=("$(printf "%-4s %-28s %-35s %-18s %-12b %-15s" "$((${#NAMES[@]}))" "$name" "$zone" "$machine" "$st" "${ip:--}")")
done <<< "$RAW"

printf "  ${BOLD}%-4s %-28s %-35s %-18s %-12s %-15s${RESET}\n" "#" "NOM" "ZONE" "MACHINE" "STATUS" "IP"
echo "  ──── ──────────────────────────── ─────────────────────────────────── ────────────────── ──────────── ───────────────"
for line in "${LINES[@]}"; do
    echo -e "  ${line}"
done

echo ""
echo -e "  ${DIM}Sélectionner les VMs à détruire (ex: ${BOLD}1,3${RESET}${DIM} ou ${BOLD}all${RESET}${DIM})${RESET}"
echo ""

while true; do
    read -rp "  Choix [1-${#NAMES[@]}/all/q]: " input
    input=$(echo "$input" | tr -d ' ')

    [[ "$input" == "q" ]] && { echo "  Annulé."; exit 0; }

    SELECTED=()
    if [[ "$input" == "all" ]]; then
        for i in "${!NAMES[@]}"; do SELECTED+=("$i"); done
        break
    fi

    valid=true
    IFS=',' read -ra parts <<< "$input"
    for p in "${parts[@]}"; do
        if [[ "$p" =~ ^[0-9]+$ ]] && [ "$p" -ge 1 ] && [ "$p" -le ${#NAMES[@]} ]; then
            SELECTED+=("$((p-1))")
        else
            echo -e "  ${RED}Invalide: $p${RESET}"; valid=false; break
        fi
    done
    $valid && [ ${#SELECTED[@]} -gt 0 ] && break
done

# Confirmation
echo ""
echo -e "  ${RED}${BOLD}VMs à détruire :${RESET}"
for idx in "${SELECTED[@]}"; do
    echo -e "    • ${NAMES[$idx]}  (${ZONES[$idx]})"
done
echo ""
read -rp "  Confirmer la suppression ? [y/N] " confirm
[[ "$confirm" != [yY] ]] && { echo "  Annulé."; exit 0; }

echo ""
for idx in "${SELECTED[@]}"; do
    name="${NAMES[$idx]}"
    zone="${ZONES[$idx]}"
    echo -ne "  Suppression de ${BOLD}${name}${RESET}..."
    if gcloud compute instances delete "$name" --zone="$zone" --quiet 2>/dev/null; then
        echo -e " ${GREEN}OK${RESET}"
    else
        echo -e " ${RED}ÉCHEC${RESET}"
    fi
done

echo -e "\n  ${GREEN}${BOLD}Terminé.${RESET}\n"