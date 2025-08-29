import lightgbm as lgb
try:
    # Intentamos crear un clasificador pasándole el parámetro para usar la GPU
    lgb.LGBMClassifier(device='gpu')
    print('LightGBM con soporte para GPU está instalado y accesible.')
except Exception as e:
    print(f'Hubo un error. Esto puede significar que la instalación falló o hay un problema con los drivers de NVIDIA: {e}')

