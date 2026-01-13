# WoofWatch Testing Guide

Guide de test complet pour valider le système de détection avant la Phase 3 (API).

## Prérequis

- Raspberry Pi 5 avec Camera Module 3 installée et activée
- Modèle YOLOv8 téléchargé (via `python scripts/download_model.py`)
- Environnement virtuel activé (`source venv/bin/activate`)

## Tests Progressifs

### Étape 1 : Validation Caméra (30 secondes)

**Objectif :** Confirmer que la Camera Module 3 fonctionne correctement avec Picamera2.

```bash
python scripts/test_camera_only.py --duration 30
```

**Critères de succès :**
- ✅ Flux vidéo fluide s'affiche
- ✅ FPS ≥ 25
- ✅ Résolution correcte (640x480 par défaut)
- ✅ Message "PASS: Camera performance excellent"

**En cas d'échec :**
1. Vérifier connexion caméra : `libcamera-hello`
2. Activer caméra : `sudo raspi-config` → Interface → Camera
3. Redémarrer le Pi

---

### Étape 2 : Validation Détection (60 secondes)

**Objectif :** Confirmer que YOLOv8 détecte correctement les chiens.

```bash
python scripts/test_detection.py --duration 60
```

**Critères de succès :**
- ✅ Bounding boxes s'affichent autour des chiens
- ✅ FPS ≥ 10 (processing FPS)
- ✅ Détections cohérentes (pas de faux positifs excessifs)
- ✅ Temps d'inférence < 150ms

**En cas de FPS trop faible (<10) :**
```bash
# Tester avec résolution plus basse
python scripts/test_detection.py --duration 60 --resolution 416x416
```

---

### Étape 3 : Optimisation avec Frame-Skip (60 secondes)

**Objectif :** Améliorer les FPS en ne traitant pas chaque frame.

```bash
# Tester différents niveaux de frame-skip
python scripts/test_detection.py --frame-skip 1 --duration 60
python scripts/test_detection.py --frame-skip 2 --duration 60
python scripts/test_detection.py --frame-skip 3 --duration 60
```

**Analyse des résultats :**
- `skip=0` : Meilleure réactivité, FPS plus bas
- `skip=1` : Bon compromis (traite 1 frame sur 2)
- `skip=2` : FPS élevés, réactivité acceptable (traite 1 frame sur 3)
- `skip=3` : FPS maximum, risque de latence perceptible

**Recommandation :** Choisir le frame-skip qui maintient FPS ≥ 20 tout en gardant détections fluides.

---

### Étape 4 : Benchmark Complet (2-4 minutes)

**Objectif :** Tester toutes les configurations et trouver les paramètres optimaux.

```bash
python scripts/benchmark_detector.py --duration 30
```

**Résultats attendus :**
```
Resolution   | Skip | Cap FPS  | Proc FPS  | Inf (ms)     | Detects
--------------------------------------------------------------------------
416x416      | 2    |    28.45 |     27.89 |   85.2 (82-91) |      15
640x480      | 1    |    25.12 |     24.67 |  102.5 (95-115) |      18
640x480      | 0    |    12.34 |     12.01 |  101.8 (98-110) |      20
```

**Recommandations du script :**
- **Best FPS** : Configuration avec le plus haut processing FPS
- **Best quality** : Configuration sans frame-skip avec meilleur temps d'inférence

---

## Configurations Recommandées

### Configuration Équilibrée (Recommandée)
```yaml
# config/settings.yaml
camera:
  resolution:
    width: 640
    height: 480
  fps: 30

detection:
  confidence_threshold: 0.5

processing:
  frame_skip: 2  # Traite 1 frame sur 3
```

**Performance attendue :** ~20-25 FPS

---

### Configuration Haute Performance
```yaml
# Pour FPS maximum
camera:
  resolution:
    width: 416
    height: 416
  fps: 30

detection:
  confidence_threshold: 0.6  # Plus strict = moins de calculs

processing:
  frame_skip: 2
```

**Performance attendue :** ~30+ FPS

---

### Configuration Haute Qualité
```yaml
# Pour meilleure précision
camera:
  resolution:
    width: 640
    height: 480
  fps: 30

detection:
  confidence_threshold: 0.4  # Plus sensible

processing:
  frame_skip: 0  # Traite toutes les frames
```

**Performance attendue :** ~10-15 FPS

---

## Mode Benchmark Détaillé

Pour des métriques plus précises :

```bash
python scripts/test_detection.py --benchmark --duration 120
```

**Métriques supplémentaires :**
- Temps d'inférence : moyenne, min, max, médiane, écart-type
- Distribution des temps d'inférence
- Détections par frame

---

## Tests en Mode Headless (Sans Affichage)

Pour tester sur un Pi sans écran ou via SSH :

```bash
# Camera seule
python scripts/test_camera_only.py --no-display --duration 30

# Détection
python scripts/test_detection.py --no-display --duration 60

# Benchmark
python scripts/benchmark_detector.py --no-display --duration 30
```

**Note :** Les snapshots sont toujours sauvegardés dans `test_snapshots/`.

---

## Analyse des Résultats

### Snapshots Sauvegardés

Tous les tests sauvegardent des snapshots dans `test_snapshots/` :

```bash
# Voir les snapshots
ls -lh test_snapshots/

# Voir les derniers snapshots
ls -lht test_snapshots/ | head -10

# Copier vers votre machine (si SSH)
scp pi@raspberrypi.local:~/woofwatch/test_snapshots/*.jpg ./
```

### Interpréter les FPS

**Capture FPS :** Frames capturées par seconde par la caméra
- Devrait être proche du FPS configuré (30 par défaut)
- Si < 25, problème hardware ou configuration

**Processing FPS :** Frames traitées par seconde par le détecteur
- Dépend de la résolution, frame-skip et complexité de la scène
- Cible : ≥ 10 pour usage réel, ≥ 20 pour expérience fluide

**Rapport Capture/Processing :**
- Si égaux : Toutes les frames sont traitées (frame_skip = 0)
- Si Processing = Capture / 2 : frame_skip = 1 (1 frame sur 2)
- Si Processing = Capture / 3 : frame_skip = 2 (1 frame sur 3)

---

## Résolution des Problèmes

### FPS trop faibles (<10)

**Solutions par ordre de priorité :**
1. Activer frame-skip : `--frame-skip 2`
2. Réduire résolution : `--resolution 416x416`
3. Augmenter seuil de confiance : `--confidence 0.6`
4. Vérifier CPU : `htop` (pas d'autres processus gourmands)
5. Vérifier température : `vcgencmd measure_temp` (<80°C)

### Détections incorrectes

**Faux positifs (détecte des objets qui ne sont pas des chiens) :**
- Augmenter confidence : `--confidence 0.6` ou `0.7`
- Vérifier éclairage (éviter contre-jour)

**Faux négatifs (ne détecte pas les chiens) :**
- Diminuer confidence : `--confidence 0.3` ou `0.4`
- Vérifier distance caméra/chiens (optimale : 2-5m)
- Tester avec frame-skip = 0 pour éliminer timing

### Latence perceptible

**Symptômes :** Détections avec retard visible
- Réduire frame-skip : essayer `--frame-skip 1` au lieu de 2
- Vérifier queue size dans config (devrait être petit)

---

## Tests Avancés

### Test de Durabilité (Long-Running)

```bash
# Test de 10 minutes
python scripts/test_detection.py --duration 600 --no-display

# Vérifier stabilité (pas de crash, FPS constant)
```

### Test de Stress (Scènes Complexes)

```bash
# Test avec seuil très bas (détecte plus d'objets)
python scripts/test_detection.py --confidence 0.2 --duration 60

# Vérifier que FPS reste acceptable
```

### Test de Différents Éclairages

```bash
# Jour
python scripts/test_detection.py --duration 30 --save-all

# Nuit (avec lumière)
python scripts/test_detection.py --duration 30 --save-all

# Comparer les snapshots
```

---

## Checklist avant Phase 3 (API)

Avant de passer au développement de l'API FastAPI, vérifier :

- [ ] Camera test réussi (FPS ≥ 25)
- [ ] Detection test réussi (FPS ≥ 10)
- [ ] Benchmark effectué
- [ ] Configuration optimale identifiée et notée
- [ ] Snapshots de test vérifiés (détections correctes)
- [ ] Tests headless réussis (pour déploiement serveur)
- [ ] Aucun crash ou erreur pendant tests prolongés

**Configuration retenue à documenter :**
```
Résolution : _______________
Frame-skip : _______________
Confidence : _______________
FPS moyen  : _______________
```

---

## Prochaine Étape : Phase 3

Une fois les tests validés, vous êtes prêt pour :
- Implémenter API FastAPI
- WebSocket streaming
- Endpoints REST
- Intégration frontend React

Les modules core (`CameraManager`, `DogDetector`) sont maintenant validés et prêts à être wrappés dans l'API.

---

## Support

En cas de problème :
1. Vérifier logs : `logs/woofwatch.log`
2. Tester caméra : `libcamera-hello`
3. Vérifier modèle : `ls -lh models/yolov8n.onnx`
4. Consulter README.md section Troubleshooting
5. Consulter CLAUDE.md pour détails techniques
