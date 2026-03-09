#!/bin/bash

VM_NAME="ml-research-montreal"
REGION="northamerica-northeast1"
ZONES=("a" "b" "c")

# Configuration des paires (Machine, GPU, Image)
# Le L4 est sur G2, le T4 est sur N1.
CONFIGS=(
    "g2-standard-4:nvidia-l4:ubuntu-accelerator-2404-amd64-with-nvidia-580-v20260225"
    "n1-standard-4:nvidia-tesla-t4:ubuntu-accelerator-2404-amd64-with-nvidia-580-v20260225"
)

IMAGE_PROJECT="ubuntu-os-accelerator-images"

echo -e "\033[1;34mLancement de la recherche de ressources GPU à Montréal...\033[0m"

while true; do
    for CONF in "${CONFIGS[@]}"; do
        # On découpe la config
        IFS=':' read -r M_TYPE G_TYPE IMAGE_NAME <<< "$CONF"
        
        for SUFFIX in "${ZONES[@]}"; do
            ZONE="${REGION}-${SUFFIX}"
            echo -e "\n\033[1;33m[ESSAI] $M_TYPE + $G_TYPE dans $ZONE\033[0m"

            gcloud compute instances create $VM_NAME \
                --zone=$ZONE \
                --machine-type=$M_TYPE \
                --accelerator=type=$G_TYPE,count=1 \
                --image=$IMAGE_NAME \
                --image-project=$IMAGE_PROJECT \
                --boot-disk-size=100GB \
                --maintenance-policy=TERMINATE \
                --scopes=https://www.googleapis.com/auth/cloud-platform \
                --quiet # Évite les questions interactives pendant la boucle

            if [ $? -eq 0 ]; then
                echo -e "\n\033[1;32m[SUCCÈS] VM créée avec succès !\033[0m"
                echo "Machine: $M_TYPE | GPU: $G_TYPE | Zone: $ZONE"
                exit 0
            fi
            
            echo "Échec de la combinaison. Passage à la suivante..."
        done
    done

    echo -e "\n\033[1;31m[ALERTE] Aucune ressource disponible à Montréal.\033[0m"
    echo "Nouvelle tentative globale dans 90 secondes (Ctrl+C pour arrêter)..."
    sleep 90
done
