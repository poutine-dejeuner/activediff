function launch-gpu
    echo "🔍 Récupération de vos quotas GPU actifs..."

    # 1. Extraire les quotas GPU non nuls de votre projet
    set -l gpu_list (gcloud compute project-info describe --format="json" | \
        jq -r '.quotas[] | select(.metric | contains("nvidia") and contains("limit") and .limit > 0) | .metric' | \
        sed 's/_per_group_gpu_limit//g; s/nvidia_//g; s/_//g' | sort -u)

    if test -z "$gpu_list"
        echo "❌ Aucun quota GPU trouvé dans ce projet."
        return 1
    end

    # 2. Définir les coûts approximatifs (Prix On-Demand 2026)
    # On crée une table de correspondance simple
    function _get_cost
        switch $argv[1]
            case t4; echo "0.35"
            case l4; echo "0.70"
            case a100; echo "2.10"
            case a10080gb; echo "3.90"
            case h100; echo "10.00"
            case *; echo "N/A"
        end
    end

    # 3. Préparer le menu pour fzf
    set -l menu_items
    for gpu in $gpu_list
        set -l cost (_get_cost $gpu)
        set menu_items $menu_items "$gpu ($cost \$/h)"
    end

    # 4. Sélection interactive
    set -l selection (printf "%s\n" $menu_items | fzf --header "Choisissez un GPU à instancier" --height 15% --reverse)
    
    if test -z "$selection"
        echo "Annulé."
        return
    end

    set -l selected_gpu (echo $selection | cut -d' ' -f1)
    set -l gpu_type "nvidia-testing-$selected_gpu" # Nom technique gcloud
    # Correction pour les noms réels gcloud (ex: nvidia-tesla-t4)
    switch $selected_gpu
        case t4; set gpu_type "nvidia-tesla-t4"
        case l4; set gpu_type "nvidia-l4"
        case a100; set gpu_type "nvidia-tesla-a100"
        case a10080gb; set gpu_type "nvidia-a100-80gb"
    end

    # 5. Liste des zones par proximité (Montréal -> Toronto -> USA East)
    set -l zones "northamerica-northeast1-a" "northamerica-northeast1-b" "northamerica-northeast1-c" \
                 "northamerica-northeast2-a" "northamerica-northeast2-b" "northamerica-northeast2-c" \
                 "us-east1-b" "us-east1-c" "us-east4-a"

    echo "🚀 Tentative d'instanciation pour $selected_gpu..."

    for zone in $zones
        echo "📡 Test dans la zone : $zone"
        
        # On tente de créer une machine g2 (pour L4) ou n1/a2 selon le GPU
        set -l machine_type "n1-standard-4"
        if test "$selected_gpu" = "l4"
            set machine_type "g2-standard-4"
        else if test "$selected_gpu" = "a100"
            set machine_type "a2-highgpu-1g"
        end

        gcloud compute instances create "research-auto-$selected_gpu" \
            --zone=$zone \
            --machine-type=$machine_type \
            --accelerator=type=$gpu_type,count=1 \
            --maintenance-policy=TERMINATE \
            --image-family=debian-11 \
            --image-project=debian-cloud \
            --boot-disk-size=50GB \
            --metadata="install-nvidia-driver=True" 2>/dev/null

        if test $status -eq 0
            echo "✅ Succès ! Machine créée dans $zone."
            return 0
        else
            echo "⚠️  Échec dans $zone (pas de stock ou quota zone)."
        end
    end

    echo "❌ Échec : Aucune zone proche n'a de $selected_gpu disponible actuellement."
end
