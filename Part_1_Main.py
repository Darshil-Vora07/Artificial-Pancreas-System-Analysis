#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np


# In[33]:


CGM_data=pd.read_csv('CGMData.csv' , usecols=['Date','Time','Sensor Glucose (mg/dL)'])
Insulin_data=pd.read_csv('InsulinData.csv', low_memory=False)


# In[34]:


CGM_data['date_time_stamp']=pd.to_datetime(CGM_data['Date'] + ' ' + CGM_data['Time'])


# In[35]:


dates_remove= CGM_data[CGM_data['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()


# In[36]:


CGM_data=CGM_data.set_index('Date').drop(index=dates_remove).reset_index()


# In[37]:


cgm_testing = CGM_data.copy()


# In[38]:


cgm_testing=cgm_testing.set_index(pd.DatetimeIndex(CGM_data['date_time_stamp']))


# In[39]:


Insulin_data['date_time_stamp']=pd.to_datetime(Insulin_data['Date'] + ' ' + Insulin_data['Time'])


# In[40]:


starting_auto=Insulin_data.sort_values(by='date_time_stamp',ascending=True).loc[Insulin_data['Alarm']=='AUTO MODE ACTIVE PLGM OFF'].iloc[0]['date_time_stamp']


# In[41]:


auto_mode_df=CGM_data.sort_values(by='date_time_stamp',ascending=True).loc[CGM_data['date_time_stamp']>=starting_auto]


# In[42]:


manual_mode_df=CGM_data.sort_values(by='date_time_stamp',ascending=True).loc[CGM_data['date_time_stamp']<starting_auto]


# In[43]:


auto_mode_df_date_index=auto_mode_df.copy()


# In[44]:


auto_mode_df_date_index=auto_mode_df_date_index.set_index('date_time_stamp')


# In[45]:


new_list=auto_mode_df_date_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()


# In[46]:


auto_mode_df_date_index=auto_mode_df_date_index.loc[auto_mode_df_date_index['Date'].isin(new_list)]


# In[48]:


hyperglycemia_critical_wholeday_automode=(auto_mode_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[49]:


hyperglycemia_critical_daytime_automode=(auto_mode_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[50]:


hyperglycemia_critical_overnight_automode=(auto_mode_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[51]:


hyperglycemia_wholeday_automode=(auto_mode_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[52]:


hyperglycemia_daytime_automode=(auto_mode_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[53]:


hyperglycemia_overnight_automode=(auto_mode_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[54]:


range_wholeday_automode=(auto_mode_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_df_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[55]:


range_daytime_automode=(auto_mode_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_df_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[56]:


range_overnight_automode=(auto_mode_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_df_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[57]:


range_sec_wholeday_automode=(auto_mode_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_df_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[58]:


range_sec_daytime_automode=(auto_mode_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_df_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[59]:


range_sec_overnight_automode=(auto_mode_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_df_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[60]:


hypoglycemia_lv1_wholeday_automode=(auto_mode_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[61]:


hypoglycemia_lv1_daytime_automode=(auto_mode_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[62]:


hypoglycemia_lv1_overnight_automode=(auto_mode_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[63]:


hypoglycemia_lv2_wholeday_automode=(auto_mode_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[64]:


hypoglycemia_lv2_daytime_automode=(auto_mode_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[65]:


hypoglycemia_lv2_overnight_automode=(auto_mode_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_df_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[67]:


manual_mode_df_index=manual_mode_df.copy()
manual_mode_df_index=manual_mode_df_index.set_index('date_time_stamp')


# In[69]:


list2=manual_mode_df_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()


# In[70]:


manual_mode_df_index=manual_mode_df_index.loc[manual_mode_df_index['Date'].isin(list2)]


# In[71]:


hyperglycemia_wholeday_manual=(manual_mode_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[72]:


hyperglycemia_daytime_manual=(manual_mode_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[73]:


hyperglycemia_overnight_manual=(manual_mode_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[74]:


hyperglycemia_critical_wholeday_manual=(manual_mode_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[75]:


hyperglycemia_critical_daytime_manual=(manual_mode_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[76]:


hyperglycemia_critical_overnight_manual=(manual_mode_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[77]:


range_wholeday_manual=(manual_mode_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)



# In[78]:


range_daytime_manual=(manual_mode_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[79]:


range_overnight_manual=(manual_mode_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[80]:


range_sec_wholeday_manual=(manual_mode_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[81]:


range_sec_daytime_manual=(manual_mode_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)



# In[82]:


range_sec_overnight_manual=(manual_mode_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[83]:


hypoglycemia_lv1_wholeday_manual=(manual_mode_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[84]:


hypoglycemia_lv1_daytime_manual=(manual_mode_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[85]:


hypoglycemia_lv1_overnight_manual=(manual_mode_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[86]:


hypoglycemia_lv2_wholeday_manual=(manual_mode_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)



# In[87]:


hypoglycemia_lv2_daytime_manual=(manual_mode_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[88]:


hypoglycemia_lv2_overnight_manual=(manual_mode_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[89]:


results_df = pd.DataFrame({'percent_time_in_hyperglycemia_overnight':[ hyperglycemia_overnight_manual.mean(axis=0),hyperglycemia_overnight_automode.mean(axis=0)],


'percent_time_in_hyperglycemia_critical_overnight':[ hyperglycemia_critical_overnight_manual.mean(axis=0),hyperglycemia_critical_overnight_automode.mean(axis=0)],


'percent_time_in_range_overnight':[ range_overnight_manual.mean(axis=0),range_overnight_automode.mean(axis=0)],


'percent_time_in_range_sec_overnight':[ range_sec_overnight_manual.mean(axis=0),range_sec_overnight_automode.mean(axis=0)],


'percent_time_in_hypoglycemia_lv1_overnight':[ hypoglycemia_lv1_overnight_manual.mean(axis=0),hypoglycemia_lv1_overnight_automode.mean(axis=0)],


'percent_time_in_hypoglycemia_lv2_overnight':[ np.nan_to_num(hypoglycemia_lv2_overnight_manual.mean(axis=0)),hypoglycemia_lv2_overnight_automode.mean(axis=0)],
                           'percent_time_in_hyperglycemia_daytime':[ hyperglycemia_daytime_manual.mean(axis=0),hyperglycemia_daytime_automode.mean(axis=0)],
                           'percent_time_in_hyperglycemia_critical_daytime':[ hyperglycemia_critical_daytime_manual.mean(axis=0),hyperglycemia_critical_daytime_automode.mean(axis=0)],
                           'percent_time_in_range_daytime':[ range_daytime_manual.mean(axis=0),range_daytime_automode.mean(axis=0)],
                           'percent_time_in_range_sec_daytime':[ range_sec_daytime_manual.mean(axis=0),range_sec_daytime_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv1_daytime':[ hypoglycemia_lv1_daytime_manual.mean(axis=0),hypoglycemia_lv1_daytime_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv2_daytime':[ hypoglycemia_lv2_daytime_manual.mean(axis=0),hypoglycemia_lv2_daytime_automode.mean(axis=0)],

                           
                           'percent_time_in_hyperglycemia_wholeday':[ hyperglycemia_wholeday_manual.mean(axis=0),hyperglycemia_wholeday_automode.mean(axis=0)],
                           'percent_time_in_hyperglycemia_critical_wholeday':[ hyperglycemia_critical_wholeday_manual.mean(axis=0),hyperglycemia_critical_wholeday_automode.mean(axis=0)],
                           'percent_time_in_range_wholeday':[ range_wholeday_manual.mean(axis=0),range_wholeday_automode.mean(axis=0)],
                           'percent_time_in_range_sec_wholeday':[ range_sec_wholeday_manual.mean(axis=0),range_sec_wholeday_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv1_wholeday':[ hypoglycemia_lv1_wholeday_manual.mean(axis=0),hypoglycemia_lv1_wholeday_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv2_wholeday':[ hypoglycemia_lv2_wholeday_manual.mean(axis=0),hypoglycemia_lv2_wholeday_automode.mean(axis=0)]
                    
                          
},
                          index=['manual_mode','auto_mode'])


# In[90]:


results_df.to_csv('Results.csv',header=False,index=False)


# In[ ]:




