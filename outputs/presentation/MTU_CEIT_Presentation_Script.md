# MTU CEIT Student Performance Prediction System — Presentation Script

Audience: CEIT project supervisor / defense panel  
Language: Burmese presentation script with English technical terms  

## Slide 1 — Student Performance Prediction & Academic Support System

မင်္ဂလာပါ ဆရာ၊ ဆရာမများခင်ဗျာ။ ဒီနေ့တင်ပြမယ့် project က Mandalay Technological University ရဲ့ Computer Engineering and Information Technology — CEIT ဌာနအတွက် ရည်ရွယ်ထားတဲ့ Student Performance Prediction and Academic Support System ဖြစ်ပါတယ်။

ဒီ system ရဲ့ အဓိကရည်ရွယ်ချက်က ကျောင်းသားတစ်ယောက်ကို “အောင်မလား၊ ကျမလား” လို့ အလိုအလျောက်ဆုံးဖြတ်ပေးဖို့ မဟုတ်ပါဘူး။ Attendance၊ previous grade၊ study time၊ participation၊ homework completion စတဲ့ အချက်အလက်တွေကို စုစည်းပြီး အကူအညီလိုအပ်နိုင်တဲ့ ကျောင်းသားကို စောစောသတိပေးနိုင်ဖို့ ဖြစ်ပါတယ်။ ANN model ကို အဓိက model အဖြစ်တည်ဆောက်ထားပြီး Logistic Regression၊ Random Forest တို့နဲ့ comparison လုပ်ထားပါတယ်။ Web application ကတော့ ဌာနတွင်းဆရာ၊ ဆရာမတွေက student record ထည့်ခြင်း၊ prediction လုပ်ခြင်း၊ intervention မှတ်တမ်းတင်ခြင်းနဲ့ report ကြည့်ခြင်းကို တစ်နေရာတည်းမှာ ဆောင်ရွက်နိုင်အောင် တည်ဆောက်ထားပါတယ်။

## Slide 2 — Why this system matters

ပထမဆုံး problem statement ကိုရှင်းပြပါမယ်။ ပညာရေးအဖွဲ့အစည်းတွေမှာ student risk signal တွေ မရှိတာမဟုတ်ပါဘူး။ Attendance ကျလာတာ၊ absence များလာတာ၊ homework မပြီးတာ၊ previous grade ကျတာတို့က final result မထွက်ခင်မှာပဲ တွေ့ရနိုင်ပါတယ်။ ပြဿနာက ဒီ signal တွေကို စနစ်တကျ စုပြီး အချိန်မီသုံးနိုင်တဲ့ workflow မရှိခြင်းပါ။

ဒါကြောင့် project objective ကို သုံးပိုင်းခွဲထားပါတယ်။ ပထမ၊ student data ကို standardized format နဲ့သိမ်းမယ်။ ဒုတိယ၊ ANN model က pass probability နဲ့ risk level ထုတ်ပေးမယ်။ တတိယ၊ teacher က result ကို review လုပ်ပြီး academic counseling၊ attendance follow-up၊ study plan စတဲ့ intervention တစ်ခုကို မှတ်တမ်းတင်နိုင်မယ်။ ဒီနေရာမှာ prediction ကို disciplinary decision အတွက် တစ်ခုတည်းသောအထောက်အထားအဖြစ် မသုံးရပါဘူး။ Human oversight ပါဝင်မှသာ responsible system ဖြစ်ပါတယ်။

## Slide 3 — CEIT-first project scope

ဒီ project ကို university တစ်ခုလုံးအတွက် စပြီးမလုပ်ဘဲ CEIT major တစ်ခုတည်းအတွက် pilot အနေနဲ့ scope ချထားပါတယ်။ အကြောင်းရင်းက major တစ်ခုအတွင်းမှာ curriculum၊ assessment policy နဲ့ teacher workflow က ပိုတူညီတာကြောင့် model နဲ့ system ကို ထိန်းချုပ်စမ်းသပ်ဖို့ လွယ်ကူပါတယ်။

Academic level ကို First Year၊ Second Year၊ Third Year၊ Fourth Year၊ Fifth Year First Semester၊ Fifth Year Second Semester၊ Final Year ဆိုပြီး ခုနစ်မျိုးထားပါတယ်။ Year ခြောက်နှစ်ရှိပေမယ့် fifth year ကို semester နှစ်ခုခွဲထားတာကြောင့် operational level ခုနစ်ခုဖြစ်ပါတယ်။ Current version ရဲ့ user တွေက CEIT ဌာနတွင်းက teacher နဲ့ administrator ဖြစ်ပါတယ်။ Real MTU student data မရသေးတဲ့အတွက် synthetic project data နဲ့ workflow ကို ပြသထားပါတယ်။ Real deployment အဆင့်မဟုတ်သေးဘဲ departmental prototype/decision-support pilot အဆင့်ဖြစ်ကြောင်း တိတိကျကျပြောရပါမယ်။

## Slide 4 — Complete system architecture

System architecture ကို ဘယ်ဘက်ကနေညာဘက်အထိ ရှင်းပြပါမယ်။ ပထမ layer က data input ဖြစ်ပြီး teacher form၊ CSV batch upload သို့မဟုတ် project synthetic dataset ကလာနိုင်ပါတယ်။ ဒုတိယ layer မှာ schema validation နဲ့ range validation လုပ်ပါတယ်။ ဥပမာ attendance ကို zero ကနေ one hundred အတွင်း၊ grade ကို သတ်မှတ်ထားတဲ့ range အတွင်းဖြစ်ကြောင်း စစ်ပါတယ်။

တတိယ layer က ML service ဖြစ်ပါတယ်။ Raw value ကို preprocessing pipeline က numerical scaling နဲ့ categorical encoding လုပ်ပြီး ANN model ဆီပို့ပါတယ်။ Model က probability ထုတ်ပေးပြီး threshold 0.5 အရ pass/fail label ပြောင်းပါတယ်။ စတုတ္ထ layer က SQLite database ဖြစ်ပြီး student record၊ prediction၊ intervention နဲ့ audit event ကို သိမ်းပါတယ်။ နောက်ဆုံး layer မှာ teacher က result ကို review လုပ်ပြီး support action ဆောင်ရွက်ပါတယ်။ Layer အားလုံးကို authentication၊ role permission၊ validation၊ rate limiting၊ audit logging နဲ့ model artifact integrity check တို့က ဝိုင်းကာကွယ်ထားပါတယ်။

## Slide 5 — Three-stage data strategy

Dataset strategy ကို သုံးဆင့်ခွဲထားပါတယ်။ ပထမဆင့်မှာ UCI Student Performance Dataset ကို official experiment reference အဖြစ်သုံးပါတယ်။ ဒီ dataset က public benchmark ဖြစ်လို့ preprocessing နဲ့ comparison procedure ကို ပြန်လည်စမ်းသပ်နိုင်ပါတယ်။ ဒုတိယဆင့်မှာ CEIT structure နဲ့ကိုက်ညီတဲ့ synthetic dataset record 1,400 ကို generate လုပ်ပါတယ်။ ဒါဟာ web app၊ database၊ training pipeline နဲ့ report workflow အားလုံးကို end-to-end စမ်းသပ်ဖို့ ဖြစ်ပါတယ်။

တတိယဆင့်က အနာဂတ်မှာ ခွင့်ပြုချက်ရရှိထားတဲ့ MTU data နဲ့ validation လုပ်ရမယ့်အဆင့်ပါ။ လက်ရှိ synthetic result ကို “MTU ကျောင်းသားတွေအပေါ် တကယ်အလုပ်လုပ်တယ်” ဆိုတဲ့ institutional evidence အဖြစ် မပြောပါဘူး။ Dataset provenance ကို metadata ထဲမှာ Synthetic MTU CEIT Project Dataset နဲ့ synthetic_data=true လို့သိမ်းထားတာက result ကို မမှားယွင်းဖော်ပြစေရန်ဖြစ်ပါတယ်။

## Slide 6 — How the synthetic CEIT dataset was created

Real dataset မရနိုင်တဲ့အခြေအနေမှာ random rows သက်သက်မဖန်တီးဘဲ controlled synthetic generation လုပ်ထားပါတယ်။ Academic level ခုနစ်ခုစီအတွက် record နှစ်ရာ၊ စုစုပေါင်း 1,400 ဖန်တီးထားပါတယ်။ Random seed ကို 20260831 သတ်မှတ်ထားလို့ dataset ကို ပြန် generate လုပ်ရင် reproducible ဖြစ်ပါတယ်။ Class distribution က pass 968 နဲ့ fail 432 ဖြစ်ပြီး imbalance နည်းနည်းရှိပါတယ်။

Grade generation logic မှာ previous grade၊ attendance၊ study time၊ participation၊ homework completion တို့က positive contribution ပေးပြီး absences နဲ့ academic difficulty က negative contribution ပေးပါတယ်။ Real life uncertainty ကို represent လုပ်ဖို့ random noise ထည့်ပါတယ်။ Gender ကို outcome equation ထဲ လုံးဝမသုံးထားပါဘူး။ Family support နဲ့ internet access က grade ကို တိုက်ရိုက်မပြောင်းဘဲ study behavior/attendance pattern ကိုသာ သက်ရောက်စေထားပါတယ်။ CSV ထဲမှာ PII မပါ၊ duplicate မရှိ၊ missing value မရှိကြောင်း validation test လုပ်ထားပါတယ်။ ဒီ dataset က demonstration data ဖြစ်ပြီး real-student truth မဟုတ်ကြောင်း presentation မှာ ထပ်မံအသိပေးရပါမယ်။

## Slide 7 — Synthetic outcome logic

ဒီ slide က synthetic dataset မှာ target ကို ဘယ်လိုဖန်တီးထားလဲရှင်းပြတာပါ။ Final grade ကို rule တစ်ခုတည်းနဲ့မဟုတ်ဘဲ weighted signal အဖြစ်တည်ဆောက်ထားပါတယ်။ Prior achievement ဖြစ်တဲ့ previous grade က အရေးကြီးတဲ့ signal ဖြစ်ပါတယ်။ Engagement group ထဲက attendance၊ study time၊ participation နဲ့ homework completion က grade ကိုတိုးစေပါတယ်။ Absences နဲ့ level difficulty က grade ကိုလျော့စေပါတယ်။ ထို့ပြင် student outcome ကို လုံးဝ deterministic မဖြစ်စေဖို့ controlled random noise ထည့်ထားပါတယ်။

Generate ပြီးတဲ့ final grade ကို 50 နှင့်အထက်ဆို pass၊ 50 အောက်ဆို fail လို့ target label သတ်မှတ်ပါတယ်။ သတိထားရမယ့်အချက်က ANN ကို ဒီ formula ကို တိုက်ရိုက်ပေးမထားပါဘူး။ ANN က generated records တွေရဲ့ input-output relationship ကို training data ကနေ သင်ယူရတာဖြစ်ပါတယ်။ ဒါကြောင့် training pipeline၊ split နဲ့ evaluation အဆင့်တွေက လိုအပ်နေဆဲဖြစ်ပါတယ်။

## Slide 8 — Preprocessing without data leakage

Training မစခင် preprocessing ကို အစဉ်လိုက်လုပ်ပါတယ်။ ပထမဆုံး column name နဲ့ category value တွေကို canonical format ပြောင်းပါတယ်။ Boolean input တွေကို yes/no သို့မဟုတ် 1/0 တစ်မျိုးတည်းဖြစ်အောင် normalize လုပ်ပြီး numerical feature တွေရဲ့ range ကို validate လုပ်ပါတယ်။ Target က pass/fail ဖြစ်ပြီး pass threshold 50 ကို သုံးထားပါတယ်။

အရေးကြီးဆုံးအချက်က dataset ကို split လုပ်ပြီးမှ preprocessor ကို train set ပေါ်မှာပဲ fit လုပ်ခြင်းပါ။ Numerical feature တွေကို StandardScaler သဘောတရားနဲ့ scale လုပ်ပြီး categorical feature တွေကို encoded values ပြောင်းပါတယ်။ Validation နဲ့ test data က train preprocessor ရဲ့ stored mean၊ scale နဲ့ mapping ကိုပဲအသုံးပြုပါတယ်။ Test set ကို fit လုပ်ရာမှာ ထည့်သုံးမိရင် data leakage ဖြစ်ပြီး accuracy မမှန်မကန်မြင့်သွားနိုင်ပါတယ်။ Preprocessor ကို JSON artifact အဖြစ် model နဲ့အတူသိမ်းထားတာကြောင့် web prediction အချိန်မှာ training တုန်းက rule အတိအကျကို ပြန်သုံးနိုင်ပါတယ်။

## Slide 9 — Reproducible train / validation / test split

Dataset 1,400 rows ကို stratified split နဲ့ခွဲပါတယ်။ ပထမ 20 percent — record 280 — ကို final test set အဖြစ်သီးသန့်ဖယ်ထားပါတယ်။ ကျန် 1,120 ထဲက 20 percent ကို validation အဖြစ်ယူတာကြောင့် validation 224 နဲ့ training 896 ရရှိပါတယ်။ Total percentage အရ train 64 percent၊ validation 16 percent၊ test 20 percent ဖြစ်ပါတယ်။

Stratified ဆိုတာ pass/fail class ratio ကို subset တစ်ခုစီမှာ အနီးစပ်ဆုံးတူအောင်ခွဲတာပါ။ Random seed 42 ကို fix လုပ်ထားပါတယ်။ ANN၊ Logistic Regression နဲ့ Random Forest သုံးမျိုးလုံးကို တူညီတဲ့ rows ပေါ်မှာ train/test လုပ်ထားလို့ comparison က fair ဖြစ်ပါတယ်။ Validation set ကို ANN ရဲ့ early stopping နဲ့ hyperparameter monitoring အတွက်သုံးပြီး test set ကို model training ပြီးဆုံးမှ တစ်ကြိမ်သာ evaluate လုပ်ပါတယ်။

## Slide 10 — ANN architecture

ANN architecture က input layer၊ hidden layer နှစ်ခုနဲ့ output layer တစ်ခုပါဝင်ပါတယ်။ Input က raw feature ဆယ်ခုဖြစ်ပေမယ့် categorical encoding ပြီးနောက် actual vector dimension က preprocessor mapping အပေါ် မူတည်ပါတယ်။ ပထမ hidden layer မှာ neuron 64၊ ဒုတိယ hidden layer မှာ neuron 32 သုံးထားပါတယ်။ Hidden layers နှစ်ခုလုံးမှာ ReLU activation သုံးပါတယ်။ ReLU က negative value ကို zero ထားပြီး positive value ကိုဖြတ်ပေးတာကြောင့် non-linear relationship ကို သင်ယူနိုင်ပါတယ်။

Overfitting လျော့ဖို့ L2 regularization နဲ့ dropout ထည့်ထားပါတယ်။ Output layer က neuron တစ်လုံးနဲ့ sigmoid activation ဖြစ်ပါတယ်။ Sigmoid output က zero နဲ့ one ကြား probability ဖြစ်ပြီး pass probability အဖြစ်အဓိပ္ပာယ်ဖော်ပါတယ်။ Decision threshold 0.5 သတ်မှတ်ထားပြီး 0.5 နှင့်အထက်ကို pass၊ အောက်ကို fail/risk review လို့ label ချပါတယ်။ ဒီ architecture က tabular dataset အတွက် အလွန်ကြီးမားမနေဘဲ project requirement ဖြစ်တဲ့ 64–32 hidden layers ကို အတိအကျလိုက်နာထားပါတယ်။

## Slide 11 — Exactly how the ANN was trained

ANN training process ကိုအသေးစိတ်ရှင်းပြပါမယ်။ Model ကို compile လုပ်ရာမှာ Adam optimizer နဲ့ binary cross-entropy loss သုံးပါတယ်။ Binary classification ဖြစ်လို့ true label နဲ့ predicted probability ကွာဟမှုကို binary cross-entropy ကတိုင်းတာပါတယ်။ Batch size 32 ဆိုတာ training rows ကို တစ်ကြိမ်လျှင် 32 rows စီ forward pass နဲ့ backpropagation လုပ်တာပါ။

Batch တစ်ခုစီမှာ input ကို network ထဲဖြတ်ပြီး probability ထုတ်ပါတယ်။ ပြီးနောက် loss တွက်၊ gradient ကို backpropagate လုပ်၊ Adam က weight နဲ့ bias ကို update လုပ်ပါတယ်။ Epoch တစ်ခုပြီးတိုင်း validation loss ကိုစစ်ပါတယ်။ Validation loss မတိုးတက်တော့ရင် early stopping က training ကိုရပ်ပြီး best validation epoch ရဲ့ weights ကိုပြန်ယူပါတယ်။ Learning improvement နှေးလာရင် ReduceLROnPlateau သဘောတရားနဲ့ learning rate လျှော့ပါတယ်။ Class distribution က 968 pass နှင့် 432 fail ဖြစ်လို့ minority fail class ကို model မမေ့စေရန် class weight သုံးပါတယ်။ Seed 42 fix လုပ်ထားတာကြောင့် split နဲ့ initialization variation ကိုတတ်နိုင်သမျှ reproducible ဖြစ်စေပါတယ်။

## Slide 12 — ANN compared with two baselines

ANN result ကို standalone မပြဘဲ Logistic Regression နဲ့ Random Forest တို့နဲ့ တူညီတဲ့ split ပေါ်မှာနှိုင်းယှဉ်ထားပါတယ်။ Synthetic CEIT benchmark မှာ ANN accuracy 0.771၊ precision 0.892၊ recall 0.763၊ F1 0.822 နဲ့ ROC-AUC 0.880 ရပါတယ်။ Logistic Regression က accuracy၊ precision၊ recall နဲ့ F1 မှာ ANN နဲ့တူပြီး AUC အနည်းငယ်သာပိုပါတယ်။ Random Forest က accuracy 0.793 နဲ့ F1 0.846 ရလို့ F1 အရအကောင်းဆုံးဖြစ်ပါတယ်။

ဒီ result က အရေးကြီးတဲ့ discussion တစ်ခုကိုပေးပါတယ်။ ANN သုံးထားတာကြောင့် အလိုအလျောက်အကောင်းဆုံးဖြစ်မသွားပါဘူး။ Dataset သေးပြီး structured tabular data ဖြစ်ရင် tree-based model သို့မဟုတ် linear baseline ကလည်းကောင်းနိုင်ပါတယ်။ Project objective က ANN ကို မဖြစ်မနေအနိုင်ပေးဖို့မဟုတ်ဘဲ reproducible comparison နဲ့ model choice ကိုသက်သေပြဖို့ဖြစ်ပါတယ်။ Current operational artifact ကို ANN အဖြစ်ထားပေမယ့် future real-data validation ပြီးရင် model selection ကို metrics၊ fairness နဲ့ explainability ပေါ်အခြေခံပြီး ပြန်ဆုံးဖြတ်သင့်ပါတယ်။

## Slide 13 — Operational ANN evaluation

Web application မှာသုံးထားတဲ့ operational ANN artifact ရဲ့ held-out test result ကိုပြထားပါတယ်။ Accuracy 76.1 percent၊ precision 89.0 percent၊ recall 74.7 percent၊ F1-score 81.2 percent ဖြစ်ပါတယ်။ Majority class ကိုပဲအမြဲခန့်မှန်းမယ်ဆိုရင် baseline accuracy 69.3 percent ပဲရမှာဖြစ်လို့ ANN က baseline ထက်ကောင်းပါတယ်။

Confusion matrix ကို fail=negative၊ pass=positive အနေနဲ့ဖတ်ပါတယ်။ True fail 68၊ false pass 18၊ false fail 49၊ true pass 145 ဖြစ်ပါတယ်။ Academic support အမြင်နဲ့ false pass 18 က အထူးသတိထားရမယ့် case ဖြစ်ပါတယ်။ တကယ် fail ဖြစ်မယ့် student ကို pass လို့မှားခန့်မှန်းရင် support လွတ်သွားနိုင်ပါတယ်။ False fail 49 ကတော့ support review ပိုလုပ်ရနိုင်ပေမယ့် teacher review နဲ့ပြန်စစ်နိုင်ပါတယ်။ ဒါကြောင့် accuracy တစ်ခုတည်းကိုမကြည့်ဘဲ precision၊ recall၊ F1 နဲ့ confusion matrix ကိုအတူတကွစဉ်းစားထားပါတယ်။

## Slide 14 — Bias audit: gender & family support

Protected or sensitive attributes ကြောင့် model behavior ကွာခြားမှုရှိမရှိ စစ်ဖို့ gender နဲ့ family support group audit လုပ်ထားပါတယ်။ ANN ရဲ့ gender accuracy gap က 2.8 percentage points၊ positive prediction rate gap 6.7 points၊ recall gap 1.4 points ဖြစ်ပါတယ်။ Family support မှာ accuracy gap 5.6 points၊ positive prediction rate gap 13.4 points၊ recall gap 8.5 points ဖြစ်ပါတယ်။

Gap ရှိတယ်ဆိုတာ discrimination ကိုတစ်ခါတည်းသက်သေပြတာမဟုတ်သလို gap သေးတယ်ဆိုတာ fairness အာမခံတာလည်းမဟုတ်ပါဘူး။ ဒီ dataset က synthetic ဖြစ်တဲ့အတွက် observed gap က generator correlation၊ sample size နဲ့ model decision boundary ကလာနိုင်ပါတယ်။ Gender ကို grade generation formula မှာမသုံးထားပေမယ့် correlated behaviors ကြောင့် model output gap ဖြစ်နိုင်ပါတယ်။ Responsible use အတွက် group-wise metric ကို real approved data ပေါ်မှာပြန်စစ်ရမယ်၊ threshold နဲ့ feature policy ကို review လုပ်ရမယ်၊ sensitive attribute ကို punitive decision အတွက်မသုံးရပါဘူး။

## Slide 15 — External UCI experiment

UCI Student Performance Dataset ကို external benchmark အဖြစ်သုံးထားပါတယ်။ Mathematics subset မှာ record 395 ရှိပြီး final grade G3 ကို threshold 10 ဖြင့် pass/fail ပြောင်းထားပါတယ်။ တူညီတဲ့ experiment framework နဲ့ ANN၊ Logistic Regression၊ Random Forest ကိုနှိုင်းယှဉ်ရာ ANN accuracy 83.5 percent၊ F1 0.869; Logistic accuracy 84.8 percent၊ F1 0.878; Random Forest accuracy 87.3 percent၊ F1 0.900 ရပါတယ်။

ဒီ result က code pipeline က public dataset ပေါ်မှာလည်း အလုပ်လုပ်ကြောင်း reproducibility evidence ပေးပါတယ်။ ဒါပေမယ့် UCI dataset က Portugal secondary-school context ဖြစ်ပြီး MTU CEIT context မဟုတ်ပါဘူး။ ထို့ကြောင့် UCI accuracy ကို MTU accuracy အဖြစ်ပြောလို့မရပါဘူး။ External benchmark၊ synthetic CEIT system test နဲ့ future approved MTU validation ကို သီးခြား evidence layers အဖြစ် ခွဲပြထားတာက project ရဲ့ research honesty အရေးကြီးတဲ့အပိုင်းဖြစ်ပါတယ်။

## Slide 16 — Web application: teacher-facing functions

Web application က teacher workflow ကို page အလိုက်ခွဲထားပါတယ်။ Login ဝင်ပြီး Overview မှာ total students၊ at-risk count၊ recent prediction နဲ့ intervention status တွေကိုကြည့်နိုင်ပါတယ်။ Students page မှာ student record create၊ search၊ edit လုပ်နိုင်ပါတယ်။ Prediction page မှာ input feature ဆယ်ခုထည့်ပြီး probability၊ predicted label နဲ့ risk indication ရရှိပါတယ်။ Interventions page မှာ counseling၊ attendance follow-up၊ tutoring စတဲ့ support action ကို assign လုပ်ပြီး status update လုပ်နိုင်ပါတယ်။ Model and Reports page မှာ metrics၊ confusion matrix၊ model comparison၊ fairness audit နဲ့ dataset provenance ကိုကြည့်နိုင်ပါတယ်။ Users page က administrator အတွက်သာဖြစ်ပါတယ်။

“Do not link to record” ဆိုတာ error မဟုတ်ပါဘူး။ Ad-hoc prediction mode ဖြစ်ပါတယ်။ Teacher က student master record မဖန်တီးချင်သေးဘဲ hypothetical case သို့မဟုတ် one-time assessment စမ်းချင်ရင် prediction ကို student record နဲ့မချိတ်ဘဲ run လို့ရပါတယ်။ Student follow-up history လိုအပ်ရင် link to record ကိုရွေးသင့်ပြီး demonstration သို့မဟုတ် privacy-sensitive check အတွက် ad-hoc ကိုသုံးနိုင်ပါတယ်။

## Slide 17 — Security, privacy & auditability

System ကို secure ဖြစ်အောင် authentication မှာ raw password မသိမ်းဘဲ PBKDF2 algorithm၊ random salt နဲ့ iterative hashing သုံးထားပါတယ်။ Login အောင်မြင်ရင် session token ရပေမယ့် database ထဲမှာ token ကို plain text မသိမ်းဘဲ hash ပဲသိမ်းထားပါတယ်။ Session lifetime ကို 8 hours သတ်မှတ်ထားပါတယ်။ Role-based access မှာ teacher က academic functions သုံးနိုင်ပြီး user account management ကို admin ပဲလုပ်နိုင်ပါတယ်။

API input ကို schema နဲ့ range validation စစ်ပြီး authentication/login endpoint တွေမှာ rate limiting ထည့်ထားပါတယ်။ Activity audit trail က ဘယ်သူ၊ ဘယ်အချိန်၊ ဘာလုပ်ခဲ့လဲကိုစစ်ဆေးနိုင်စေပါတယ်။ Model၊ preprocessor နဲ့ metadata artifacts ကို manifest ထဲက SHA-256 checksum နဲ့စစ်လို့ file ပြောင်းလဲမှုကို detect လုပ်နိုင်ပါတယ်။ API ကို local 127.0.0.1 မှာ run လုပ်ထားပြီး CORS allowlist ကိုကန့်သတ်ထားပါတယ်။ Production deployment လုပ်မယ်ဆို HTTPS၊ managed database၊ encrypted backups၊ secret manager နဲ့ institutional data policy ကိုထပ်ထည့်ရပါမယ်။

## Slide 18 — Conclusion & next validation steps

အဆုံးသတ်အနေနဲ့ လက်ရှိ project မှာ CEIT-specific synthetic dataset၊ reproducible preprocessing and training pipeline၊ 64–32 ANN architecture၊ Logistic Regression/Random Forest comparison၊ gender/family support bias audit၊ secure teacher web workflow၊ intervention tracking နဲ့ model reporting အားလုံးပြီးစီးထားပါတယ်။ ဒါကြောင့် simple prediction notebook တစ်ခုထက်ပိုပြီး complete decision-support prototype တစ်ခုဖြစ်ပါတယ်။

ဒါပေမယ့် real institutional system ဖြစ်ဖို့ အရေးကြီးတဲ့နောက်တစ်ဆင့်တွေရှိပါတယ်။ Formal permission ရပြီး de-identified MTU CEIT data ကိုကာလအလိုက်စုစည်းရပါမယ်။ Cohort နဲ့ semester တစ်ခုကို train လုပ်ပြီး နောက် cohort ကို test လုပ်တဲ့ temporal validation လိုပါတယ်။ Probability calibration၊ fairness group audit၊ threshold policy နဲ့ model drift monitoring ကိုပြန်စစ်ရပါမယ်။ Teacher standard operating procedure၊ student notification၊ correction/appeal process နဲ့ data retention policy လည်းသတ်မှတ်ရပါမယ်။ နောက်ဆုံးအချက်က ဒီ system ရဲ့ prediction ဟာ teacher ကိုစောစောကူညီပေးတဲ့ signal ဖြစ်ပြီး teacher ရဲ့ professional judgment ကို အစားမထိုးပါဘူး။ ကျေးဇူးတင်ပါတယ်။ မေးခွန်းများရှိရင် ဖြေကြားပြီး live demonstration ဆက်လုပ်ပါမယ်။

