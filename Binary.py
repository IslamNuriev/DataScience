import numpy as np
import pandas as pd
import seaborn as sns
import copy
import statistics
import os
import matplotlib.pyplot as plt
from eli5.permutation_importance import get_score_importances
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from category_encoders import CatBoostEncoder
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest   
from sklearn.svm import OneClassSVM
from scipy.stats import spearmanr
from tqdm import tqdm, tqdm_notebook
import lightgbm as lgb
import xgboost as xgb
from IPython.display import display, Image
import time
import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve
from sklearn.metrics import mean_absolute_error, recall_score, precision_score, roc_auc_score, average_precision_score,f1_score
#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_log_error,mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import cross_val_predict as cvp

class Validation_Binary():
    
    """Класс валидации моделей бинарной классификации
    Параметры:
    ---------------
    target: string
        Имя целевой переменной
    ---------------
    features: list
        Список факторов, на которых обучается модель
    ---------------
    train: DataFrame
        Обучающая выборка (включает в себя features, target,keys,dt_field)
    ---------------
    test: DataFrame
        Тестовая выборка (включает в себя features, target,keys,dt_field)
    ---------------
    model_obj:
        Модель с параметрами (обучается внутри класса)
    ---------------
    main_metric: string (default: 'ROC AUC')
        Ключевая метрика. Возможные значения: 'ROC AUC','PR AUC','F1','Precision','Recall'
    ---------------
    make_output_dir: boolean (default: True)
        Необходимость создания директории с результатами валидации
    ---------------
    model_id: int or str (default: 777)
        ID модели
    ---------------
    threshold: float (default: 0.5)
        Пороговый уровень прогноза (если меньше, то 0, если больше, то 1)
    ---------------
    permute: boolean (default: True)
        Необходимость расчёта Permutation Importance для train и test (на больших выборках занимает много времени)
    ---------------
    output_dir: str (default: 'results_{}'.format(model_id))
        Имя директории с результатами валидации
    ---------------
    folds: tuple (default: None)
        Кортеж со списками индексов тренировочных и тестовых фолдов 
    ---------------
    keys: str (default: None)
        Имя переменной первичного ключа 
    ---------------
    dt_field: str (default: None)
        Имя переменной времени
    ---------------
    cat_features: list (default: [])
        Список категориальных переменных (нужен в Тесте 1.7 и Тесте 1.10 при очистке от выбросов)
     
    """
    
     # Экземпляр класса
    def __init__(self,
                 target,
                 features:list,
                 train,
                 test,
                 model_obj, # Модель
                 main_metric='ROC AUC',
                 make_output_dir=True,
                 model_id='binary_777',
                 threshold=0.5,
                 permute=True,
                 output_dir=None,
                 folds=None,
                 keys=None,
                 dt_field=None,
                 cat_features=[],
                 ):
            
        if folds is not None:
            self._folds=folds
        else:
            self._folds=None
            
        self._train=train
        self._test=test
        self._dt_field = dt_field
        self._model_obj = model_obj.fit(train[features],train[target])
        self._keys = keys
        self._target=target
        self._features=features
        self._main_metric=main_metric
        self._model_id=model_id
        self._threshold=threshold
        self._cat_features=cat_features
        
        if make_output_dir==True:
            if output_dir is None:
                self._output_dir = 'results_{}'.format(model_id)
            else:
                self._output_dir = output_dir
            os.makedirs(self._output_dir, exist_ok=True)
        
        # Прогнозы
        self._train_prob=self._model_obj.predict_proba(train[features])[:,1]
        self._train_pred=np.where(self._train_prob<threshold,0,1)
        
        self._test_prob=self._model_obj.predict_proba(test[features])[:,1]
        self._test_pred=np.where(self._test_prob<threshold,0,1)
        
        if self._folds is not None:
            
            model_fold=copy.deepcopy(self._model_obj)
            self._oof_prob=cvp(model_fold , train[features],train[target], cv=folds,method='predict_proba')[:,1]
            self._oof_pred=np.where(self._oof_prob<threshold,0,1)
            
        # Словарь метрик
        metric_dict={}
        metric_dict['ROC AUC']='roc_auc'
        metric_dict['PR AUC']='average_precision'
        metric_dict['F1']='f1'
        metric_dict['Precision']='precision'
        metric_dict['Recall']='recall'
        
        self._metric_dict=metric_dict
        self._scoring=self._metric_dict[self._main_metric]
        
        # Словарь score_func
        if self._main_metric=='ROC AUC':
            def score_func(X,y):
                y_prob=self._model_obj.predict_proba(X)[:,1]
                return roc_auc_score(y,y_prob)
        
        if self._main_metric=='PR AUC':
            def score_func(X,y):
                y_prob=self._model_obj.predict_proba(X)[:,1]
                return average_precision_score(y,y_prob)
            
        if self._main_metric=='F1':
            def score_func(X,y):
                y_pred=self._model_obj.predict(X)
                return f1_score(y,y_pred)
            
        if self._main_metric=='Precision':
            def score_func(X,y):
                y_pred=self._model_obj.predict(X)
                return precision_score(y,y_pred)
            
        if self._main_metric=='Recall':
            def score_func(X,y):
                y_pred=self._model_obj.predict(X)
                return recall_score(y,y_pred)
        
        # Permutation Importance
        if permute==True:
            print('--------------------------------------------------------')
            print('Производится расчёт Permutation Importance для Train и Test')
            
            X_train=self._train[self._features]
            y_train=self._train[self._target]
            X_test=self._test[self._features]
            y_test=self._test[self._target]
            
            _, local_imp_train = get_score_importances(score_func, X_train.values, y_train.values, n_iter=5, random_state=0)
            _, local_imp_test = get_score_importances(score_func, X_test.values, y_test.values, n_iter=5, random_state=0)
            
            self._perm_imp_train = np.mean(local_imp_train, axis=0)
            self._perm_imp_test = np.mean(local_imp_test, axis=0)
            
            self.all_imp = pd.DataFrame(index=self._features)
            
            self.all_imp['PI train'] = self._perm_imp_train
            self.all_imp['PI test'] = self._perm_imp_test
            self.all_imp['PI difference'] = self._perm_imp_train - self._perm_imp_test
            
            self.all_imp.to_csv(f'results_{self._model_id}/t2_3_PI.csv')
            
    def stats(self):
        
        """Выводит количество наблюдений и уровень целевого события по месяцам для каждой из выборок (необходима переменная dt_field)"""
        
        # Входные параметры
        if self._dt_field is None:
            print('Для проведения теста необходимо ввести переменную времени dt_field')
            
        else:
            
            # Тестовая выборка
            aggs = []
            test_data = pd.DataFrame()
            test_data[self._dt_field]=self._test[self._dt_field]
            test_data[self._target]=self._test[self._target]

            test_data[self._dt_field] = pd.to_datetime(test_data[self._dt_field])
            test_data.sort_values(self._dt_field,
                             inplace=True)
            test_data[self._dt_field] = test_data[self._dt_field].dt.strftime('%Y-%m')

            # Расчёт метрик
            for name, data_grp in tqdm_notebook(test_data.groupby(self._dt_field)):

                if np.sum(data_grp[self._target]) > 0:
                    data_grp = data_grp.copy()
                else:
                    continue

                n_obs = np.shape(data_grp)[0]

                agg = {
                    f'{self._dt_field}': name,
                    'Target Rate': round(data_grp[self._target].mean(),3),
                    'N_OBS': n_obs,
                }

                aggs += [agg]
                
            aggs_dynamic = pd.DataFrame(aggs)
            aggs_dynamic.sort_values(self._dt_field,
                                     inplace=True)
            aggs_dynamic.reset_index(inplace=True,
                                     drop=True)
            
            print('Тестовая выборка')
            display(aggs_dynamic)
                
            # Обучающая выборка
            aggs = []
            train_data = pd.DataFrame()
            train_data[self._dt_field]=self._train[self._dt_field]
            train_data[self._target]=self._train[self._target]

            train_data[self._dt_field] = pd.to_datetime(train_data[self._dt_field])
            train_data.sort_values(self._dt_field,
                             inplace=True)
            train_data[self._dt_field] = train_data[self._dt_field].dt.strftime('%Y-%m')

            # Расчёт метрик
            for name, data_grp in tqdm_notebook(train_data.groupby(self._dt_field)):

                if np.sum(data_grp[self._target]) > 0:
                    data_grp = data_grp.copy()
                else:
                    continue

                n_obs = np.shape(data_grp)[0]

                agg = {
                    f'{self._dt_field}': name,
                    'Target Rate': round(data_grp[self._target].mean(),3),
                    'N_OBS': n_obs,
                }

                aggs += [agg]
                

            aggs_dynamic = pd.DataFrame(aggs)
            aggs_dynamic.sort_values(self._dt_field,
                                     inplace=True)
            aggs_dynamic.reset_index(inplace=True,
                                     drop=True)
            
            print('Обучающая выборка')
            display(aggs_dynamic)


        #return aggs_dynamic

        
#-------------------------------РАЗДЕЛ 1. КОЛИЧЕСТВЕННЫЕ ТЕСТЫ--------------------------------------------------------------------------

#     def t1_1_nans(self,chosen_samples: list = None):
        
#         if chosen_samples is not None:
#             chosen_samples = chosen_samples
#         else:
#             chosen_samples = self._samples        
        
#         for s in chosen_samples:
#             sample=self._samples[s]
#             labels =self._model_obj.predict(sample)

    def t1_2_quality(self):
        
        """Выводит значения 5-ти метрик качества, ставит светофор по ключевой метрике и строит кривые ROC и PR"""
                
        
        metrics = []
        col_list=[]
        target=self._target
        
        sem_dict={
                  'ROC AUC':[0.6,0.7],
                  'PR AUC':[0.1,0.2],
                  'F1':[0.5,0.7],
                  'Precision':[0.5,0.7],
                  'Recall':[0.5,0.7]
        }
        
                
        st = {
              'ROC AUC': round(roc_auc_score(self._train[target],self._train_prob),4),
              'PR AUC': round(average_precision_score(self._train[target],self._train_prob),4),
              'F1': round(f1_score(self._train[target],self._train_pred),4),
              'Precision': round(precision_score(self._train[target],self._train_pred),4),
              'Recall': round(recall_score(self._train[target],self._train_pred),4)
               }

        metrics += [st]
        col_list.append('Train')
                            
        st = {
              'ROC AUC': round(roc_auc_score(self._test[target],self._test_prob),4),
              'PR AUC': round(average_precision_score(self._test[target],self._test_prob),4),
              'F1': round(f1_score(self._test[target],self._test_pred),4),
              'Precision': round(precision_score(self._test[target],self._test_pred),4),
              'Recall': round(recall_score(self._test[target],self._test_pred),4),
               }

        metrics += [st]
        col_list.append('Test')
        
                    
        if self._folds is not None:
            
            st = {
                  'ROC AUC': round(roc_auc_score(self._train[target],self._oof_prob),4),
                  'PR AUC': round(average_precision_score(self._train[target],self._oof_prob),4),
                  'F1': round(f1_score(self._train[target],self._oof_pred),4),
                  'Precision': round(precision_score(self._train[target],self._oof_pred),4),
                  'Recall': round(recall_score(self._train[target],self._oof_pred),4)
                   }
            
            metrics += [st]
            col_list.append('OOF')
            
        results = pd.DataFrame(metrics).T
        results.columns = col_list
        
        self._metric_table=results
            
        display(results)
            
        # Выставление итогового светофора
        metric=results[results.index==self._main_metric]['Test'].values[0]
        self._metric=metric
            
        if metric<=sem_dict[self._main_metric][0]:
            img = 'lights/red.png'
            comment1=f'Ключевая метрика {self._main_metric} на тестовой выборке меньше {sem_dict[self._main_metric][0]},'
            comment2=f'что соответствует красному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        elif metric>=sem_dict[self._main_metric][1]:
            img = 'lights/green.png'
            comment1=f'Ключевая метрика {self._main_metric} на тестовой выборке больше {sem_dict[self._main_metric][1]},'
            comment2=f'что соответствует зелёному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        else:
            img = 'lights/yellow.png'
            comment1=f'Ключевая метрика {self._main_metric} на тестовой выборке лежит в пределах {sem_dict[self._main_metric][0]}                                                                  -{sem_dict[self._main_metric][1]},'
            comment2=f'что соответствует зелёному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        # Графики
        fig , (ax1 , ax2) = plt.subplots(1 , 2 , figsize=(15,5))
    
        fpr, tpr, _ =precision_recall_curve(self._test[target], self._test_prob)
        ax1.plot(tpr, fpr)

        fpr, tpr, _ = precision_recall_curve(self._train[target], self._train_prob)
        ax1.plot(tpr, fpr)
        
        if self._folds is not None:
            fpr, tpr, _ = precision_recall_curve(self._train[target], self._oof_prob)
            ax1.plot(tpr, fpr)
            plt.legend(['test','train','oof'])
        else:
            plt.legend(['test','train'])            

        ax1.set_title("Precision-Recall Curve")

        fpr, tpr, _ = roc_curve(self._test[target], self._test_prob)
        ax2.plot(fpr, tpr)
        
        fpr, tpr, _ = roc_curve(self._train[target], self._train_prob)
        ax2.plot(fpr, tpr)

        if self._folds is not None:
            fpr, tpr, _ = roc_curve(self._train[target], self._oof_prob)
            ax2.plot(fpr, tpr)
            plt.legend(['test','train','oof'])
        else:
            plt.legend(['test','train'])            

        ax2.set_title("ROC Curve")
        
        # Сохранение результатов
        plt.savefig(f'results_{self._model_id}/t1_2_plots.png')
        results.to_csv(f'results_{self._model_id}/t1_2_metrics.csv')
        
        
            
    def t1_3_quality_dynamic(self):
        
        """Выводит динамику ключевой метрики на тестовой выборке по месяцам (необходима переменная dt_field)"""
        
        # Входные параметры
        if self._dt_field is None:
            print('Для проведения теста необходимо ввести переменную времени (dt_field)')
        else:
            test_data = pd.DataFrame()
            test_data[self._dt_field]=self._test[self._dt_field]
            test_data[self._target]=self._test[self._target]

            test_data['Probability']=self._test_prob
            test_data['Prediction']=self._test_pred

            test_data[self._dt_field] = pd.to_datetime(test_data[self._dt_field])
            test_data.sort_values(self._dt_field,
                             inplace=True)
            test_data[self._dt_field] = test_data[self._dt_field].dt.strftime('%Y-%m')

            # Расчёт метрик
            aggs = []
            for name, data_grp in tqdm_notebook(test_data.groupby(self._dt_field)):

                if np.sum(data_grp[self._target]) > 0:
                    data_grp = data_grp.copy()
                else:
                    continue

                auc_i = roc_auc_score(y_score=data_grp['Probability'],
                                      y_true=data_grp[self._target])
                avg_pr_i = average_precision_score(y_score=data_grp['Probability'],
                                      y_true=data_grp[self._target])
                pr_i = precision_score(y_pred=data_grp['Prediction'],
                                      y_true=data_grp[self._target])
                f1_i = f1_score(y_pred=data_grp['Prediction'],
                                y_true=data_grp[self._target])
                rec_i = recall_score(y_pred=data_grp['Prediction'],
                                      y_true=data_grp[self._target])

                metric_dict={}
                metric_dict['ROC AUC']=auc_i
                metric_dict['PR AUC']=avg_pr_i
                metric_dict['F1']=f1_i
                metric_dict['Precision']=pr_i
                metric_dict['Recall']=rec_i

                n_obs = np.shape(data_grp)[0]

                agg = {
                    f'{self._dt_field}': name,
                    f'{self._main_metric}': metric_dict[self._main_metric],
                    'N_OBS': n_obs,
                }

                aggs += [agg]
            table_for_g_dyn = pd.DataFrame(aggs)

            aggs_dynamic = pd.DataFrame(aggs)
            aggs_dynamic.sort_values(self._dt_field,
                                     inplace=True)
            aggs_dynamic.reset_index(inplace=True,
                                     drop=True)

            # plot
            sns.set_style('white')
            fig, ax1 = plt.subplots(figsize=(14, 6))
            plt.ylabel('Количество наблюдений', fontdict={'fontsize': 16})
            plt.xlabel('Период наблюдений', fontdict={'fontsize': 16})

            # N_OBS
            plt.bar(np.arange(0, len(aggs_dynamic)), aggs_dynamic['N_OBS'].values,
                    color='darkseagreen', label='N_OBS', align='center')
            plt.grid(axis='y')
            ax1.legend(loc='upper left')
            plt.ylim(0, np.max(aggs_dynamic['N_OBS']) + np.std(aggs_dynamic['N_OBS']))

            # Metric
            ax2 = ax1.twinx()
            plt.ylabel(f'{self._main_metric}', fontdict={'fontsize': 16})
            plt.plot(aggs_dynamic[f'{self._main_metric}'].values, linewidth=2, linestyle='-', label=f'{self._main_metric}', marker='o', markersize=8,
                     color='green');

            plt.title(f'Динамика {self._main_metric}', fontdict={'fontsize': 18})

            if aggs_dynamic[self._dt_field].dtype == '<M8[ns]':
                plt.xticks(np.arange(0, len(aggs_dynamic)
                                     ), [str(i.year
                                             ) + '-' + str(i.month) for i in aggs_dynamic[self._dt_field]])
            else:
                plt.xticks(np.arange(0, len(aggs_dynamic)), [str(i) for i in aggs_dynamic[self._dt_field]])

            ax1.spines['left'].set_color('w')
            ax1.xaxis.set_tick_params(rotation=45)
            ax2.legend()

            plt.tight_layout()
            plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
            plt.rc('legend', fontsize=15)  # legend fontsize

            # Сохранение результатов
            plt.savefig(f'results_{self._model_id}/t1_3_dynamic.png')

            return table_for_g_dyn
        
    def t1_4_calibration_curve(self,n_bins=10,name='Model'):
        
        """Выводит кривую калибровки (тест проводится для вероятностных моделей)"""
        
        # Модель и данные
        y_test=self._test[self._target]
        y_pred=self._test_prob

        # Calibration Curve
        model_y, model_x=calibration_curve(y_test,y_pred, n_bins=n_bins)
        
        fig, ax = plt.subplots(figsize=(15,10))
        plt.plot(model_x,model_y, marker='o', linewidth=1, label=name)
        
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        fig.suptitle('Calibration plot')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('True probability in each bin')
        plt.legend()
        plt.show()
    
    def t1_5_overfitting(self,learning_curves=True):
        
        """Выводит относительную разницу метрик и ставит светофор по ключевой метрике
        Параметры:
        ---------------
        learning_curves: boolean (default: True)
            Необходимость построения кривых обучения (ставить True только для бустинго, иначе выдаст ошибку)
        """
        
        ## Сравнение метрики на трейне и тесте
        metrics=self._metric_table
        metrics['Relative Difference']=round(abs(metrics['Train']-metrics['Test'])/metrics['Train'],2)
        display(metrics)
        
        main_over=metrics[metrics.index==self._main_metric]['Relative Difference'].values[0]
        
        if main_over>=0.5:
            img = 'lights/red.png'
            comment1=f'Относительное изменение {self._main_metric} на валидации составляет более 50%,'
            comment2=f'что соответствует красному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        elif main_over<=0.3:
            img = 'lights/green.png'
            comment1=f'Относительное изменение {self._main_metric} на валидации составляет менее 50%,'
            comment2=f'что соответствует зелёному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        else:
            img = 'lights/yellow.png'
            comment1=f'Относительное изменение {self._main_metric} на валидации составляет менее 50%, но более 30%'
            comment2=f'что соответствует жёлтому валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        # Сохранение результатов
        metrics.to_csv(f'results_{self._model_id}/t1_5_metric_diff.csv')
            
        ## Кривые обучения
        if learning_curves==True:
            X_train=self._train[self._features]
            y_train=self._train[self._target]
            X_test=self._test[self._features]
            y_test=self._test[self._target]

            #X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

            eval_set=[(X_train, y_train), (X_test, y_test)]
            copy_model=copy.deepcopy(self._model_obj)
            copy_model.fit(X_train, y_train, eval_metric=["error", "logloss","auc"], eval_set=eval_set, verbose=False)

            results = copy_model.evals_result_
            epochs = len(results['training']['binary_error'])
            x_axis = range(0, epochs)

            # plot log loss
            fig, ax = plt.subplots(figsize=(15,7))
            ax.grid()
            ax.plot(x_axis, results['training']['binary_logloss'], label='Train')
            ax.plot(x_axis, results['valid_1']['binary_logloss'], label='Test')
            ax.legend()

            plt.ylabel('Log Loss',fontsize=13)
            plt.xlabel('Iterations',fontsize=13)
            plt.title('Learning Curves',fontsize=16)
            plt.savefig(f'results_{self._model_id}/t1_5_learning_curves.png')
            plt.show()
            
            
        
    def t1_6_opt_params(self,params_grid,cv):
        
        """Подбирает гиперпараметры к исходной модели
        Параметры:
        ---------------
        params_grid: dict
            Сетка гиперпараметров
        ---------------
        cv: int or fold
            Способ кросс-валидации при подборе гиперпараметров
        """
        
        X_train=self._train[self._features]
        y_train=self._train[self._target]
        X_test=self._test[self._features]
        y_test=self._test[self._target]
        
        print('Поиск параметров для модели')
        model=copy.deepcopy(self._model_obj)
        
        ## RandomizedSearchCV
        Search=RandomizedSearchCV(model,
                                    param_distributions=params_grid,
                                    n_iter=10,
                                    scoring=self._scoring,
                                    n_jobs=-1,
                                    cv=cv,
                                    verbose=0)
        
        ## Fitting Model
        Search.fit(X_train,y_train)

        ## Best Model
        best_model=copy.deepcopy(type(self._model_obj))(**Search.best_params_).fit(X_train,y_train)
        
        y_prob=best_model.predict_proba(X_test)[:,1]
        y_pred=np.where(y_prob<self._threshold,0,1)
        
        ## Score of Best Model
        auc_score=roc_auc_score(y_test,y_prob)
        avg_pr_score=average_precision_score(y_test,y_prob)
        pr_score=precision_score(y_test,y_pred)
        rec_score=recall_score(y_test,y_pred)
        f_score=f1_score(y_test,y_pred)
        
        metric_dict={}
        metric_dict['ROC AUC']=auc_score
        metric_dict['PR AUC']=avg_pr_score
        metric_dict['F1']=f_score
        metric_dict['Precision']=pr_score
        metric_dict['Recall']=rec_score
        
        Opt_score=metric_dict[self._main_metric]
        
        print(f'Лучшие альтернативные параметры: {best_model}')
        print(f'{self._main_metric} на альтернативном наборе параметров на тестовой выборке: {round(Opt_score,4)}')
        print('----------------------------------------------------------------------------------------------------')
        
        ## Выставление итогового светофора
        increase=(Opt_score-self._metric)/self._metric
        
        if increase>=0.1:
            img = 'lights/red.png'
            comment1='Альтернативный набор гиперпараметров приводит к повышению метрики качества на валидации более чем на 10% от изначального уровня качества,'
            comment2=f'что соответствует красному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        elif increase<=0.05:
            img = 'lights/green.png'
            comment1='Альтернативный набор гиперпараметров не приводит к значимому повышению метрики качества на валидации,'
            comment2=f'что соответствует зелёному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        else:
            img = 'lights/yellow.png'
            comment1=f'Альтернативный набор гиперпараметров приводит к повышению метрики качества на валидации в пределах 5-10% от изначального уровня качества,'
            comment2=f'что соответствует жёлтому валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
        
    def t1_7_interpret_models(self,params_grid_logreg,params_grid_tree,
                              cv,
                              cats_strategy='delete'):
        
        """Подбирает гиперпараметры к упрощённым моделям
        Параметры:
        ---------------
        params_grid_logreg: dict
            Сетка гиперпараметров для логистической регрессии
        ---------------
        params_grid_tree: dict
            Сетка гиперпараметров для дерева решений
        ---------------
        cv: int or fold
            Способ кросс-валидации при подборе гиперпараметров
        ---------------
        cats_strategy: str (default: None)
            Способ обработки категориальных признаков. Возможные значения: 'delete', 'encode'  
            
            Если есть категориальные переменные, то упрощённые модели воспринимают их как численные.
            
            Также упрощённые модели не работают с пропусками.
        """
        
        X_train=self._train[self._features]
        y_train=self._train[self._target]
        X_test=self._test[self._features]
        y_test=self._test[self._target]
        
        if cats_strategy=='delete':
            X_train.drop(self._cat_features,axis=1,inplace=True)
            X_test.drop(self._cat_features,axis=1,inplace=True)
            
        elif cats_strategy=='encode':
            enc = CatBoostEncoder()
            enc.fit(X_train[self._cat_features], y_train)
            X_train[self._cat_features] = enc.transform(X_train[self._cat_features]).fillna(0)
            X_test[self._cat_features] = enc.transform(X_test[self._cat_features]).fillna(0)
        
        print('Поиск параметров для логистической регрессии')
        Logistic_model = LogisticRegression()

        ## RandomizedSearchCV
        Search_Logistic=RandomizedSearchCV(Logistic_model,
                                            param_distributions=params_grid_logreg,
                                            n_iter=10,
                                            scoring=self._scoring,
                                            n_jobs=-1,
                                            cv=cv,
                                            verbose=0)
        
        ## Fitting Model
        Search_Logistic.fit(X_train.fillna(0),y_train)

        ## Best Model
        best_model=LogisticRegression(**Search_Logistic.best_params_).fit(X_train.fillna(0),y_train)
        
        y_prob=best_model.predict_proba(X_test.fillna(0))[:,1]
        y_pred=np.where(y_prob<self._threshold,0,1)
        
        ## Score of Best Model
        auc_score=roc_auc_score(y_test,y_prob)
        avg_pr_score=average_precision_score(y_test,y_prob)
        pr_score=precision_score(y_test,y_pred)
        rec_score=recall_score(y_test,y_pred)
        f_score=f1_score(y_test,y_pred)
        
        metric_dict={}
        metric_dict['ROC AUC']=auc_score
        metric_dict['PR AUC']=avg_pr_score
        metric_dict['F1']=f_score
        metric_dict['Precision']=pr_score
        metric_dict['Recall']=rec_score
        
        Log_score=metric_dict[self._main_metric]
        
        print('------------------------------------------------------------------------------------')
        print(f'Лучшая логистическая регрессия: {best_model}')
        print(f'{self._main_metric} логистической регрессии на тестовой выборке: {round(Log_score,4)}')
        print('------------------------------------------------------------------------------------')
                                
        print('Поиск параметров для дерева решений')
        Tree_model = DecisionTreeClassifier()

        ## RandomizedSearchCV
        Search_Tree=RandomizedSearchCV(Tree_model,
                                            param_distributions=params_grid_tree,
                                            n_iter=10,
                                            scoring=self._scoring,
                                            n_jobs=-1,
                                            cv=cv,
                                            verbose=0)
        
        ## Fitting Model
        Search_Tree.fit(X_train.fillna(0),y_train)

        ## Best Model                    
        best_model=DecisionTreeClassifier(**Search_Tree.best_params_).fit(X_train.fillna(0),y_train)
                                
        y_prob=best_model.predict_proba(X_test.fillna(0))[:,1]
        y_pred=np.where(y_prob<self._threshold,0,1)
        
        ## Score of Best Model
        auc_score=roc_auc_score(y_test,y_prob)
        avg_pr_score=average_precision_score(y_test,y_prob)
        pr_score=precision_score(y_test,y_pred)
        rec_score=recall_score(y_test,y_pred)
        f_score=f1_score(y_test,y_pred)
        
        metric_dict={}
        metric_dict['ROC AUC']=auc_score
        metric_dict['PR AUC']=avg_pr_score
        metric_dict['F1']=f_score
        metric_dict['Precision']=pr_score
        metric_dict['Recall']=rec_score
        
        Tree_score=metric_dict[self._main_metric]
        
        print('------------------------------------------------------------------------------------')
        print(f'Лучшее дерево решений: {best_model}')
        print(f'{self._main_metric} дерева решений на тестовой выборке: {round(Tree_score,4)}')
        print('------------------------------------------------------------------------------------')
                                 
        max_score=max(Log_score,Tree_score)
        
        ## Выбор лучшей модели
        if max_score==Log_score:
            best_model=LogisticRegression(**Search_Logistic.best_params_)
        else:
            best_model=DecisionTreeClassifier(**Search_Tree.best_params_)
            
        
        ## Выставление итогового светофора
        increase=(max_score-self._metric)/self._metric
        
        if increase>=0.05:
            img = 'lights/red.png'
            comment1='Упрощение модели возможно и метрика качества увеличивается более чем на 5%,'
            comment2=f'что соответствует красному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        elif increase<=-0.02:
            img = 'lights/green.png'
            comment1='Упрощение модели невозможно без существенного снижения метрики качества модели,'
            comment2=f'что соответствует зелёному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        else:
            img = 'lights/yellow.png'
            comment1=f'Упрощение модели возможно с незначительным снижением метрики качества,'
            comment2=f'что соответствует жёлтому валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
    def t1_8_feature_selection(self,top=7,step=5):
        
        """Строит Uplift-кривую для Forward Selection
        Параметры:
        ---------------
        top: int (default: 7)
            Число наиболее важных переменных для показа
        ---------------
        step: int (default: 5)
            Число наиболее важных переменных, которые будут добавляться каждую итерацию
        """
        
        X_train=self._train[self._features]
        y_train=self._train[self._target]
        X_test=self._test[self._features]
        y_test=self._test[self._target]
        
        # Важность признаков
        print('Оценивается важность признаков модели')
        print('----------------------------------------------------------------------------------------------------')
        
        # Обучение
        X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.25, random_state=42) 
        model_train=copy.deepcopy(self._model_obj).fit(X_tr, y_tr)
        
        # Качество
        y_prob=model_train.predict_proba(X_te)[:,1]
        y_pred=np.where(y_prob<self._threshold,0,1)
        
        auc_score=roc_auc_score(y_te,y_prob)
        avg_pr_score=average_precision_score(y_te,y_prob)
        pr_score=precision_score(y_te,y_pred)
        rec_score=recall_score(y_te,y_pred)
        f_score=f1_score(y_te,y_pred)

        metric_dict={}
        metric_dict['ROC AUC']=auc_score
        metric_dict['PR AUC']=avg_pr_score
        metric_dict['F1']=f_score
        metric_dict['Precision']=pr_score
        metric_dict['Recall']=rec_score

        base_score=metric_dict[self._main_metric]

        # Результаты
        split_train= model_train.feature_importances_
        
        data_imp=pd.DataFrame()
        data_imp['Features']=X_train.columns
        data_imp['Importances']=split_train
            
        data_imp_sorted=data_imp.sort_values(by='Importances',ascending=False)
        display(data_imp_sorted.head(top))
        
        print('----------------------------------------------------------------------------------------------------')
        print('Производится Uplift Test')
        print('----------------------------------------------------------------------------------------------------')
        
        # Отбор признаков       
        n_feat_list=[0]
        metric_list=[0]
        base_score_list=[base_score]
        
        for k in range(1,int(len(self._features)/step)+1):
            top_ind=step*k
            top_feat=data_imp_sorted['Features'].iloc[:top_ind]
            
            # Обучение           
            model_feat=copy.deepcopy(self._model_obj).fit(X_tr[top_feat], y_tr)
            
            # Качество
            y_prob=model_feat.predict_proba(X_te[top_feat])[:,1]
            y_pred=np.where(y_prob<self._threshold,0,1)
            
            auc_score=roc_auc_score(y_te,y_prob)
            avg_pr_score=average_precision_score(y_te,y_prob)
            pr_score=precision_score(y_te,y_pred)
            rec_score=recall_score(y_te,y_pred)
            f_score=f1_score(y_te,y_pred)
            
            metric_dict={}
            metric_dict['ROC AUC']=auc_score
            metric_dict['PR AUC']=avg_pr_score
            metric_dict['F1']=f_score
            metric_dict['Precision']=pr_score
            metric_dict['Recall']=rec_score

            Opt_score=metric_dict[self._main_metric]
            
            n_feat_list.append(step*k)
            metric_list.append(round(Opt_score,3))
            base_score_list.append(round(base_score,3))
            
        # Результаты
        data_forward=pd.DataFrame()
        data_forward['Top']=n_feat_list
        data_forward['Metric']=metric_list
        
        # График
        fig = plt.figure(figsize=(15,8))
        plt.ylabel(self._main_metric, fontsize=13)
        plt.xlabel('Most important k features', fontsize=13)
        plt.title('Uplift by Forward Selection', fontsize=16)
        plt.plot(n_feat_list, metric_list, linewidth=2, markersize=10, marker='s')
        plt.xticks(np.arange(0, len(metric_list)*step, step=step))
        plt.plot(n_feat_list,base_score_list, linewidth=4, linestyle=':')
        plt.legend(['Metric on k features', 'Base score'])
        
        # Сохранение результатов
        plt.savefig(f'results_{self._model_id}/t1_8_forward.png')
        
        plt.show()
            
        
    def t1_10_object_selection(self,undersampling=True,oversampling=True,outliers=True,out=0.07,
                               cats_strategy='delete',seed=42):
        
        """Оценивает возможность применения сэмплирования и удаления выбросов)
        Параметры:
        ---------------
        undersampling: boolean (default: True)
            Необходимость использования RandomUnderSampler
        ---------------
        oversampling: boolean (default: True)
            Необходимость использования RandomOverSampler
        ---------------
        outliers: boolean (default: True)
            Необходимость использования IsolationForest
        ---------------
        out: float (default: 0.07)
            Доля наблюдений, рассматриваемых как выбросы
        ---------------
        cats_strategy: str (default: None)
            Способ обработки категориальных признаков. Возможные значения: 'delete', 'encode'  
            
            Если есть категориальные переменные, то IsolationForest воспринимает их как численные.
            
            Также IsolationForest не работает с пропусками.
        """

        X_train=self._train[self._features]
        y_train=self._train[self._target]
        X_test=self._test[self._features]
        y_test=self._test[self._target]
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        min_und=round(y_tr.mean(),2)+0.01
        
#         for c in self._cat_features:
#             X_train[c]=X_train[c].fillna(str(0))
#             X_test[c]=X_test[c].fillna(str(0))

        # Undersampling
        if undersampling==True:
            print('Поиск оптимального уровня сэмплирования для RandomUnderSampling')
            print('----------------------------------------------------------------------------------------------------')

            # Подроб уровня сэмплирования
            metric_list=[]
            thr_list=[]
            
            for l in np.arange(min_und,min_und+0.071,0.01):
                undersample = RandomUnderSampler(sampling_strategy=l,random_state=seed)
                X_tr_under, y_tr_under = undersample.fit_resample(X_tr, y_tr)

                # Обучение
                model_under=copy.deepcopy(self._model_obj).fit(X_tr_under, y_tr_under)
                
                y_prob=model_under.predict_proba(X_te)[:,1]
                y_pred=np.where(y_prob<self._threshold,0,1)
                
                auc_score=roc_auc_score(y_te,y_prob)
                avg_pr_score=average_precision_score(y_te,y_prob)
                pr_score=precision_score(y_te,y_pred)
                rec_score=recall_score(y_te,y_pred)
                f_score=f1_score(y_te,y_pred)

                metric_dict={}
                metric_dict['ROC AUC']=auc_score
                metric_dict['PR AUC']=avg_pr_score
                metric_dict['F1']=f_score
                metric_dict['Precision']=pr_score
                metric_dict['Recall']=rec_score
                
                thr_list.append(l)
                metric_list.append(round(metric_dict[self._main_metric],4))
                
            # Оптимальный уровень
            opt_thr=thr_list[np.argmax(metric_list)]
            print('----------------------------------------------------------------------------------------------------')
            print(f'Оптимальный уровень сэмплирования для RandomUnderSampling: {round(opt_thr*100,2)}%')
            print('----------------------------------------------------------------------------------------------------')
            
            # Модель
            undersample = RandomUnderSampler(sampling_strategy=opt_thr,random_state=seed)
            X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
            
            model_under=copy.deepcopy(self._model_obj).fit(X_train_under, y_train_under)
            
            y_prob=model_under.predict_proba(X_test)[:,1]
            y_pred=np.where(y_prob<self._threshold,0,1)

            auc_score=roc_auc_score(y_test,y_prob)
            avg_pr_score=average_precision_score(y_test,y_prob)
            pr_score=precision_score(y_test,y_pred)
            rec_score=recall_score(y_test,y_pred)
            f_score=f1_score(y_test,y_pred)

            metric_dict={}
            metric_dict['ROC AUC']=auc_score
            metric_dict['PR AUC']=avg_pr_score
            metric_dict['F1']=f_score
            metric_dict['Precision']=pr_score
            metric_dict['Recall']=rec_score

            Opt_score=metric_dict[self._main_metric]

            print('----------------------------------------------------------------------------------------------------')
            print(f'{self._main_metric} на тестовой выборке после RandomUnderSampling: {round(Opt_score,4)}')
            
        # Oversampling
        if oversampling==True:
            print('Поиск оптимального уровня сэмплирования для RandomOverSampling')
            print('----------------------------------------------------------------------------------------------------')

            # Подроб уровня сэмплирования
            metric_list=[]
            thr_list=[]
            
            for l in np.arange(min_und,min_und+0.071,0.01):
                oversample = RandomOverSampler(sampling_strategy=l,random_state=seed)
                X_tr_over, y_tr_over = oversample.fit_resample(X_tr, y_tr)

                # Обучение
                model_over=copy.deepcopy(self._model_obj).fit(X_tr_over, y_tr_over)
                
                y_prob=model_over.predict_proba(X_te)[:,1]
                y_pred=np.where(y_prob<self._threshold,0,1)
                
                auc_score=roc_auc_score(y_te,y_prob)
                avg_pr_score=average_precision_score(y_te,y_prob)
                pr_score=precision_score(y_te,y_pred)
                rec_score=recall_score(y_te,y_pred)
                f_score=f1_score(y_te,y_pred)

                metric_dict={}
                metric_dict['ROC AUC']=auc_score
                metric_dict['PR AUC']=avg_pr_score
                metric_dict['F1']=f_score
                metric_dict['Precision']=pr_score
                metric_dict['Recall']=rec_score
                
                thr_list.append(l)
                metric_list.append(round(metric_dict[self._main_metric],4))
                
            # Оптимальный уровень
            opt_thr=thr_list[np.argmax(metric_list)]
            print('----------------------------------------------------------------------------------------------------')
            print(f'Оптимальный уровень сэмплирования для RandomOverSampling: {round(opt_thr*100,2)}%')
            print('----------------------------------------------------------------------------------------------------')
            
            # Модель
            oversample = RandomOverSampler(sampling_strategy=opt_thr,random_state=seed)
            X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
            
            model_over=copy.deepcopy(self._model_obj).fit(X_train_over, y_train_over)
            
            y_prob=model_under.predict_proba(X_test)[:,1]
            y_pred=np.where(y_prob<self._threshold,0,1)

            auc_score=roc_auc_score(y_test,y_prob)
            avg_pr_score=average_precision_score(y_test,y_prob)
            pr_score=precision_score(y_test,y_pred)
            rec_score=recall_score(y_test,y_pred)
            f_score=f1_score(y_test,y_pred)

            metric_dict={}
            metric_dict['ROC AUC']=auc_score
            metric_dict['PR AUC']=avg_pr_score
            metric_dict['F1']=f_score
            metric_dict['Precision']=pr_score
            metric_dict['Recall']=rec_score

            Opt_score=metric_dict[self._main_metric]

            print('----------------------------------------------------------------------------------------------------')
            print(f'{self._main_metric} на тестовой выборке после RandomOverSampling: {round(Opt_score,4)}')
            
        # Выбросы
        if outliers==True:

            if cats_strategy=='delete':
                X_train.drop(self._cat_features,axis=1,inplace=True)
                X_test.drop(self._cat_features,axis=1,inplace=True)

            elif cats_strategy=='encode':
                enc = CatBoostEncoder()
                enc.fit(X_train[self._cat_features], y_train)
                X_train[self._cat_features] = enc.transform(X_train[self._cat_features]).fillna(0)
                X_test[self._cat_features] = enc.transform(X_test[self._cat_features]).fillna(0)
                    
            print('----------------------------------------------------------------------------------------------------')
            print(f'Производится удаление выбросов с помощью IsolationForest на уровне {round(out*100,2)}%')
            print('----------------------------------------------------------------------------------------------------')
            
            # identify outliers in the training dataset
            iso = IsolationForest(contamination=out)
            yhat = iso.fit_predict(X_train.fillna(0))
            # select all rows that are not outliers
            mask = yhat != -1
            
            X_train=self._train[self._features]
            y_train=self._train[self._target]

            X_train_out, y_train_out = X_train[mask], y_train[mask]
            
            # Обучение
            model_out=copy.deepcopy(self._model_obj).fit(X_train_out, y_train_out)

            # Качество
            y_prob=model_out.predict_proba(X_test)[:,1]
            y_pred=np.where(y_prob<self._threshold,0,1)
            
            auc_score=roc_auc_score(y_test,y_prob)
            avg_pr_score=average_precision_score(y_test,y_prob)
            pr_score=precision_score(y_test,y_pred)
            rec_score=recall_score(y_test,y_pred)
            f_score=f1_score(y_test,y_pred)

            metric_dict={}
            metric_dict['ROC AUC']=auc_score
            metric_dict['PR AUC']=avg_pr_score
            metric_dict['F1']=f_score
            metric_dict['Precision']=pr_score
            metric_dict['Recall']=rec_score

            Opt_score=metric_dict[self._main_metric]
            
            print('----------------------------------------------------------------------------------------------------')
            print(f'{self._main_metric} на тестовой выборке после удаления выбросов: {round(Opt_score,2)}')
            
    def t2_1_PI_mean(self,top=10,figsize=(5,10)):
        
        """Оценивает Permutation Importance признаков в среднем (чтобы сработало, в классе нужно проставить: permute=True)
        Параметры:
        ---------------
        top: int (default: 10)
            Сколько наиболее важных переменных выводить
        ---------------
        figsize: tuple (default: (5,10))
            Размер графика
        """
               
        # График
        PI_data = self.all_imp.sort_values(by='PI test', ascending=False)
        _, (ax2,ax1) = plt.subplots(2,1,figsize=(5,10))

        sns.barplot(ax=ax1,x=PI_data[f'PI test'].iloc[:top], y=PI_data.index[:top], color='#0047ab')
        props = {'xlabel': 'Mean Importance',
                 'title': 'Permutation importance on Test'}
        ax1.set(**props)

        PI_data=self.all_imp.sort_values(by=f'PI train', ascending=False)
        sns.barplot(ax=ax2,x=PI_data[f'PI train'].iloc[:top], y=PI_data.index[:top], color='#0047ab')
        props = {'xlabel': 'Mean Importance',
                 'title': 'Permutation importance on Train'}
        ax2.set(**props)

        # Сохранение результатов
        plt.savefig(f'results_{self._model_id}/t2_1_PI_mean.png')
        
        plt.show()
        
        
    def t2_3_PI_diff(self,top=10):
        
        """Оценивает различия Permutation Importance признаков в среднем (чтобы сработало, в классе нужно проставить: permute=True)
        Параметры:
        ---------------
        top: int (default: 10)
            Сколько наиболее важных переменных выводить
        """
        
        PI_data=self.all_imp.sort_values(by=f'PI train', ascending=False)
        PI_data['Relative Diff']=PI_data['PI difference']/PI_data['PI train']
        
        display(PI_data.head(top))
        
        ## Выставление итогового светофора
        share=len(PI_data[PI_data['Relative Diff']>0.5])/len(PI_data)
        
        print(f'Значимость {round(share*100,1)}% признаков на валидации упала более чем на 50%')
        print('----------------------------------------------------------------------------------------------------')
        
        if share>=0.8:
            img = 'lights/red.png'
            comment1='Более 80% признаков имеют падение значимости более 50% по сравнению с тренировочными фолдами,'
            comment2=f'что соответствует красному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        elif share<=0.5:
            img = 'lights/green.png'
            comment1='Менее 50% признаков имеют падение значимости более 50% по сравнению с тренировочными фолдами,'
            comment2=f'что соответствует зелёному валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))
            
        else:
            img = 'lights/yellow.png'
            comment1=f'Более 50%, но менее 80% признаков имеют падение значимости более 50% по сравнению с тренировочными фолдами,'
            comment2=f'что соответствует жёлтому валидационному светофору'
            comment=comment1+comment2
            print(comment)
            display(Image(img))


        
                                
        
            
        
                