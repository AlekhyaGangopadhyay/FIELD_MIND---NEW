import os
import warnings
import joblib
import pandas as pd
import numpy as np

# Suppress scikit-learn feature name validation UserWarnings
warnings.simplefilter("ignore", category=UserWarning)

class Tier1Monitor:
    """
    Tier 1 continuous monitoring wrapper.
    Loads and runs all pre-trained scikit-learn models across Gas, Env, Vibration, and Ultrasonic domains.
    Implements robust fallbacks for missing keys/shapes to ensure zero runtime crashes.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.models = {}
        self.load_all_models()
        
    def load_all_models(self):
        """Loads all project models from their serialized joblib locations."""
        print("Loading Tier 1 pre-trained model registry...")
        
        # 1. Gas Models
        gas_dir = os.path.join(self.root_dir, "gas_sensors", "models")
        gas_files = {
            'methane': 'mq4_gas_classifier.joblib',
            'smoke_fire': 'smoke_fire_alarm_model.joblib',
            'lpg_cng': 'gas_hazard_lpg_cng.joblib',
            'co_nox': 'gas_hazard_co_nox_c6h6.joblib',
            'smoke_env': 'gas_hazard_smoke_env.joblib',
            'air_quality': 'air_quality_regressor.joblib'
        }
        for key, name in gas_files.items():
            path = os.path.join(gas_dir, name)
            self._load_single_model('gas_' + key, path)
            
        # 2. Environmental Models
        env_dir = os.path.join(self.root_dir, "temperature_humidity", "models")
        env_files = {
            'iforest': 'isolation_forest_iot.joblib',
            'occupancy': 'random_forest.joblib'
        }
        for key, name in env_files.items():
            path = os.path.join(env_dir, name)
            self._load_single_model('env_' + key, path)
            
        # 3. Vibration Models
        vib_dir = os.path.join(self.root_dir, "vibration", "models")
        vib_files = {
            'classifier': 'best_random_forest_classifier.joblib',
            'regressor': 'best_gradient_boosting_regressor.joblib'
        }
        for key, name in vib_files.items():
            path = os.path.join(vib_dir, name)
            self._load_single_model('vib_' + key, path)
            
        # 4. Ultrasonic Models
        ultra_dir = os.path.join(self.root_dir, "ultrasonic_sensors", "models")
        ultra_files = {
            'ultra_2': 'best_ultrasonic_2.joblib',
            'ultra_4': 'best_ultrasonic_4.joblib',
            'ultra_24': 'best_ultrasonic_24.joblib'
        }
        for key, name in ultra_files.items():
            path = os.path.join(ultra_dir, name)
            self._load_single_model(key, path)
            
    def _load_single_model(self, key, path):
        if os.path.exists(path):
            try:
                loaded = joblib.load(path)
                self.models[key] = loaded
                print(f"  Loaded model: {key:<15} from {os.path.basename(path)}")
            except Exception as e:
                print(f"  Error loading model {key} from {path}: {e}")
        else:
            print(f"  Warning: Model {key:<15} not found at {path}")

    # --- INFERENCE RUNNERS ---

    def evaluate_gas(self, gas_features):
        """
        Evaluates gas concentration inputs across multiple specialized models.
        """
        hazards = {}
        
        # A. Gas Methane (128 features)
        if 'gas_methane' in self.models:
            if isinstance(gas_features, dict) and 'mq4_features' in gas_features:
                X = np.array(gas_features['mq4_features']).reshape(1, -1)
            elif not isinstance(gas_features, dict) and len(gas_features) == 128:
                X = np.array(gas_features).reshape(1, -1)
            else:
                # Fallback to zero vector if dimension doesn't match
                X = np.zeros((1, 128))
            methane_pred = self.models['gas_methane'].predict(X)[0]
            hazards['methane_hazard'] = int(methane_pred == 1)
            
        # B. Smoke/Fire (36 features)
        if 'gas_smoke_fire' in self.models:
            if isinstance(gas_features, dict) and 'smoke_features' in gas_features:
                X = np.array(gas_features['smoke_features']).reshape(1, -1)
            elif not isinstance(gas_features, dict) and len(gas_features) == 36:
                X = np.array(gas_features).reshape(1, -1)
            else:
                X = np.zeros((1, 36))
            smoke_pred = self.models['gas_smoke_fire'].predict(X)[0]
            hazards['smoke_alarm'] = int(smoke_pred == 1)
            
        # C. LPG/CNG Hazard (2 features: MQ2_LPG_ppm, MQ4_CH4_ppm)
        if 'gas_lpg_cng' in self.models:
            if isinstance(gas_features, dict) and 'MQ2_LPG_ppm' in gas_features:
                X = np.array([gas_features['MQ2_LPG_ppm'], gas_features.get('MQ4_CH4_ppm', 0.0)]).reshape(1, -1)
            elif not isinstance(gas_features, dict) and len(gas_features) == 2:
                X = np.array(gas_features).reshape(1, -1)
            else:
                X = np.zeros((1, 2))
            lpg_pred = self.models['gas_lpg_cng'].predict(X)[0]
            hazards['lpg_hazard'] = int(lpg_pred == 1)
            
        # D. CO/NOx Hazard (3 features: MQ7_CO_ppm, MQ135_NOx_ppm, MQ3_Benzene_ppm)
        if 'gas_co_nox' in self.models:
            if isinstance(gas_features, dict) and 'MQ7_CO_ppm' in gas_features:
                X = np.array([gas_features['MQ7_CO_ppm'], gas_features.get('MQ135_NOx_ppm', 0.0), gas_features.get('MQ3_Benzene_ppm', 0.0)]).reshape(1, -1)
            elif not isinstance(gas_features, dict) and len(gas_features) == 3:
                X = np.array(gas_features).reshape(1, -1)
            else:
                X = np.zeros((1, 3))
            co_pred = self.models['gas_co_nox'].predict(X)[0]
            hazards['co_nox_hazard'] = int(co_pred == 1)
            
        # E. Smoke/Env Hazard (3 features: PM25_Dust_ugm3, Temp_C, Humidity_pct)
        if 'gas_smoke_env' in self.models:
            if isinstance(gas_features, dict) and 'PM25_Dust_ugm3' in gas_features:
                X = np.array([gas_features['PM25_Dust_ugm3'], gas_features.get('Temp_C', 0.0), gas_features.get('Humidity_pct', 0.0)]).reshape(1, -1)
            elif not isinstance(gas_features, dict) and len(gas_features) == 3:
                X = np.array(gas_features).reshape(1, -1)
            else:
                X = np.zeros((1, 3))
            se_pred = self.models['gas_smoke_env'].predict(X)[0]
            hazards['smoke_env_hazard'] = int(se_pred == 1)
            
        # F. Air Quality Regressor (7 features)
        if 'gas_air_quality' in self.models:
            if isinstance(gas_features, dict) and 'air_quality_features' in gas_features:
                X = np.array(gas_features['air_quality_features']).reshape(1, -1)
            elif not isinstance(gas_features, dict) and len(gas_features) == 7:
                X = np.array(gas_features).reshape(1, -1)
            else:
                X = np.zeros((1, 7))
            aq_score = self.models['gas_air_quality'].predict(X)[0]
            hazards['air_quality_score'] = float(aq_score)
            
        return hazards

    def evaluate_env(self, env_features):
        """
        Evaluates environmental sensors.
        """
        res = {}
        
        # A. Isolation Forest (9 features)
        if 'env_iforest' in self.models:
            if isinstance(env_features, dict) and 'temp' in env_features:
                # Construct 9 features in order
                f_list = [
                    env_features.get('temp', 0.0),
                    env_features.get('humidity', 0.0),
                    env_features.get('temp_hum_product', 0.0),
                    env_features.get('temp_hum_ratio', 0.0),
                    env_features.get('humidex', 0.0),
                    env_features.get('temp_roll_mean_5', 0.0),
                    env_features.get('temp_roll_std_5', 0.0),
                    env_features.get('humidity_roll_mean_5', 0.0),
                    env_features.get('humidity_roll_std_5', 0.0)
                ]
                X = np.array(f_list).reshape(1, -1)
            elif not isinstance(env_features, dict) and len(env_features) == 9:
                X = np.array(env_features).reshape(1, -1)
            else:
                X = np.zeros((1, 9))
                
            iforest_pred = self.models['env_iforest'].predict(X)[0]
            res['anomaly_detected'] = int(iforest_pred == -1)
            
        # B. Occupancy RF (23 features)
        if 'env_occupancy' in self.models:
            if isinstance(env_features, dict) and 'occupancy_features' in env_features:
                X = np.array(env_features['occupancy_features']).reshape(1, -1)
            elif not isinstance(env_features, dict) and len(env_features) == 23:
                X = np.array(env_features).reshape(1, -1)
            else:
                X = np.zeros((1, 23))
                
            occupancy_pred = self.models['env_occupancy'].predict(X)[0]
            res['occupancy_state'] = int(occupancy_pred)
            
        return res

    def evaluate_vibration(self, vibration_features):
        """
        Evaluates seismic/blast vibration sensor metrics.
        """
        res = {}
        
        # A. RF PPV Hazard Classifier (14 features)
        if 'vib_classifier' in self.models:
            if isinstance(vibration_features, dict) and 'offset' in vibration_features:
                f_list = [
                    vibration_features.get('offset', 0.0),
                    vibration_features.get('max_charge', 0.0),
                    vibration_features.get('total_charge', 0.0),
                    vibration_features.get('num_holes', 0.0),
                    vibration_features.get('detonator_code', 0.0),
                    vibration_features.get('trid_12', 0.0),
                    vibration_features.get('trid_13', 0.0),
                    vibration_features.get('trid_14', 0.0),
                    vibration_features.get('gx', 0.0),
                    vibration_features.get('gy', 0.0),
                    vibration_features.get('gelev', 0.0),
                    vibration_features.get('sx', 0.0),
                    vibration_features.get('sy', 0.0),
                    vibration_features.get('selev', 0.0)
                ]
                X = np.array(f_list).reshape(1, -1)
            elif not isinstance(vibration_features, dict) and len(vibration_features) == 14:
                X = np.array(vibration_features).reshape(1, -1)
            else:
                X = np.zeros((1, 14))
                
            vib_pred = self.models['vib_classifier'].predict(X)[0]
            res['vibration_hazard'] = int(vib_pred == 1)
            
        # B. Gradient Boosting PPV Regressor (17 features)
        if 'vib_regressor' in self.models:
            if isinstance(vibration_features, dict) and 'offset' in vibration_features:
                f_list = [
                    vibration_features.get('offset', 0.0),
                    vibration_features.get('max_charge', 0.0),
                    vibration_features.get('total_charge', 0.0),
                    vibration_features.get('num_holes', 0.0),
                    vibration_features.get('detonator_code', 0.0),
                    vibration_features.get('trid_12', 0.0),
                    vibration_features.get('trid_13', 0.0),
                    vibration_features.get('trid_14', 0.0),
                    vibration_features.get('gx', 0.0),
                    vibration_features.get('gy', 0.0),
                    vibration_features.get('gelev', 0.0),
                    vibration_features.get('sx', 0.0),
                    vibration_features.get('sy', 0.0),
                    vibration_features.get('selev', 0.0),
                    vibration_features.get('scaled_distance_usbm', 0.0),
                    vibration_features.get('scaled_distance_langefors', 0.0),
                    vibration_features.get('elevation_diff', 0.0)
                ]
                X = np.array(f_list).reshape(1, -1)
            elif not isinstance(vibration_features, dict) and len(vibration_features) == 17:
                X = np.array(vibration_features).reshape(1, -1)
            else:
                X = np.zeros((1, 17))
                
            log_ppv_pred = self.models['vib_regressor'].predict(X)[0]
            res['predicted_ppv'] = float(np.exp(log_ppv_pred))
            
        return res

    def evaluate_ultrasonic(self, ultrasonic_features, config_type=24):
        """
        Evaluates ultrasonic sensors to predict robot steering decisions.
        """
        model_key = f'ultra_{config_type}'
        if model_key not in self.models:
            return {}
            
        expected_len = config_type
        
        if isinstance(ultrasonic_features, dict):
            if config_type == 2:
                features_vector = [ultrasonic_features.get('SD_front', 0.0), ultrasonic_features.get('SD_left', 0.0)]
            elif config_type == 4:
                features_vector = [
                    ultrasonic_features.get('SD_front', 0.0), ultrasonic_features.get('SD_left', 0.0),
                    ultrasonic_features.get('SD_right', 0.0), ultrasonic_features.get('SD_back', 0.0)
                ]
            else:
                features_vector = [ultrasonic_features.get(f'US{i}', 0.0) for i in range(1, 25)]
        elif not isinstance(ultrasonic_features, dict) and len(ultrasonic_features) == expected_len:
            features_vector = list(ultrasonic_features)
        else:
            features_vector = [0.0] * expected_len
            
        X = np.array(features_vector).reshape(1, -1)
        res = {}
        
        model_data = self.models[model_key]
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            classes_map = model_data.get('classes', {})
        else:
            model = model_data
            classes_map = {}
            
        pred_encoded = model.predict(X)[0]
        pred_label = classes_map.get(int(pred_encoded), str(pred_encoded))
        
        res['steering_decision'] = pred_label
        res['sharp_turn_required'] = int(pred_label == 'Sharp-Right-Turn')
        
        return res
