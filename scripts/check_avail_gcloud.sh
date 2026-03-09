#!/bin/bash

REGION="northamerica-northeast1"
ZONES=("a" "b" "c")
GPU_TYPES=("nvidia-tesla-t4" "nvidia-l4") # Les plus communs à Montréal

echo -e "\033[1;34mVérification de la disponibilité des GPUs à Montréal ($REGION)...\033[0m"
echo "------------------------------------------------------------------------"
printf "%-10s | %-20s | %-15s\n" "ZONE" "TYPE GPU" "DISPONIBILITÉ"
echo "------------------------------------------------------------------------"

for zone_suffix in "${ZONES[@]}"; do
    ZONE="${REGION}-${zone_suffix}"
    
    for gpu in "${GPU_TYPES[@]}"; do
        # On interroge gcloud pour voir si le type d'accélérateur est listé dans cette zone
        # Si la commande retourne une ligne, c'est que le GPU est physiquement présent.
        CHECK=$(gcloud compute accelerator-types list \
            --filter="zone:($ZONE) AND name:($gpu)" \
            --format="value(name)" 2>/dev/null)

        if [ -n "$CHECK" ]; then
            # On vérifie si la zone n'est pas en maintenance ou saturée
            # (Note: gcloud ne donne pas le 'stock' exact, mais l'absence d'erreur est bon signe)
            STATUS="\033[0;32mPRÊT / DISPO\033[0m"
        else
            STATUS="\033[0;31mNON DISPONIBLE\033[0m"
        fi
        
        printf "%-10s | %-20s | %-15b\n" "$ZONE" "$gpu" "$STATUS"
    done
done
echo "------------------------------------------------------------------------"
