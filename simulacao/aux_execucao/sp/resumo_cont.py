import json, sys
d = json.load(open(sys.argv[1]))
print("quadros usados", d["quadros_usados_por_drone"], "totais", d["quadros_totais_por_drone"])
if "deteccao_bruta" in d:
    print("AP", {k: round(v["ap"], 3) for k, v in d["deteccao_bruta"]["ap"].items()})
    for tau, v in d["deteccao_bruta"]["por_limiar"].items():
        print("tau", tau, {k: (round(x["precisao"], 3), round(x["revocacao"], 3), round(x["mae_contagem_quadro"], 2), round(x["vies_contagem_quadro"], 2)) for k, x in v.items()})
for m in ("botsort", "botsort_pose", "botsort_semgmc", "chao"):
    if m in d:
        av = d[m]["avaliacao_contagem"]; an = d[m]["analise_clusters"]; pr = d[m]["deteccao_por_quadro_trilhas"]
        print(m, "unica", av["contagem_proposta"], "vistas", av["pessoas_vistas_gabarito"], "cena", av["pessoas_na_cena"],
              "corretos", av["clusters_corretos_unicos"], "dup", av["clusters_duplicados"], "falsos", av["clusters_falsos"], "perdidas", av["pessoas_perdidas"],
              "| dup entre drones", an["duplicatas_entre_drones"], "mesmo", an["duplicatas_mesmo_drone"], "na banda", an["duplicatas_na_banda"],
              "multi-drone", an["clusters_com_mais_de_um_drone"], "| soma det", d[m]["agregacao"].get("soma_deteccoes"),
              "sem posse", d[m]["agregacao"].get("faixas_sem_posse"), "cd", av["outras_estrategias"]["cerise_drones_agregacao"].get("proposto_posse_dedup_borda"),
              "ids brutos", d[m]["agregacao"].get("ids_brutos_botsort"))
        print("   trilhas P/R/MAE", {k: (round(x["precisao"], 3), round(x["revocacao"], 3), round(x["mae_contagem_quadro"], 2)) for k, x in pr.items()})
print("tempo", d.get("tempo_deteccao"), d.get("tempo_total_s"))
