if (max_label >= 1) and (max_label <= 90):
    label_predict = 'whatsapp'
elif (max_label >= 91) and (max_label <= 180):
    label_predict = 'linkedin'
elif (max_label >= 181) and (max_label <= 270):
    label_predict = 'tokopedia'
elif (max_label >= 271) and (max_label <= 360):
    label_predict = 'gmail'
elif (max_label >= 361) and (max_label <= 450):
    label_predict = 'powerpoint'
elif (max_label >= 451) and (max_label <= 540):
    label_predict = 'word'

if (max_label >= 1) and (max_label <= 45):
    label_predict = 'whatsapp'
elif (max_label >= 46) and (max_label <= 90):
    label_predict = 'linkedin'
elif (max_label >= 91) and (max_label <= 135):
    label_predict = 'tokopedia'
elif (max_label >= 136) and (max_label <= 180):
    label_predict = 'gmail'
elif (max_label >= 181) and (max_label <= 225):
    label_predict = 'powerpoint'
elif (max_label >= 226) and (max_label <= 270):
    label_predict = 'word'

whatsapp_match_value=[]
    whatsapp_unmatch_value = []
    linkedin_match_value=[]
    linkedin_unmatch_value = []
    tokopedia_match_value = []
    tokopedia_unmatch_value = []
    gmail_match_value = []
    gmail_unmatch_value = []
    powerpoint_match_value = []
    powerpoint_unmatch_value = []
    word_match_value = []
    word_unmatch_value = []

signal = signal.flatten()
        mean,freq = dp.spectral_statistics(emphasizedSignal, rate)
        #mean2,freq2 = dp.spectral_statistics(signal, rate)

#print("rata-rata freq(Hz) : ", mean)

        data_latih = 'code/umum/data_train/_ (' + str(max_label) + ').wav'
        emphasizedSignalz, signalz, ratez = preEmphasis(data_latih)
        signalz = signalz.flatten()
        filtz = lpc(emphasizedSignalz, order=16)
        lpc_featurez = filtz.numerator[1:]
        lpc_refeaturez = np.reshape(lpc_featurez, (-1, 1))  # eshape to matrix
        meanz,freqz = dp.spectral_statistics(emphasizedSignalz, ratez)
        #mean2z,freq2z = dp.spectral_statistics(signalz, ratez)
        print("Max freq(Hz) uji : ", max(freq))
        print("max freq(Hz) latih : ", max(freqz))
        print("selisih freq(Hz) latih : ", max(freq)-max(freqz))
        print("leght freq(Hz) uji : ", len(freq))
        print("leght freq(Hz) latih : ", len(freqz))
        print("selisih leght freq(Hz) latih : ", len(freq) - len(freqz))
        print("jumlah freq(Hz) latih : ", sum(freq)/len(freq))
        print("jumlah freq(Hz) latih : ", sum(freqz)/len(freqz))
        print("selisih leght freq(Hz) latih : ", (sum(freq)/len(freq)) - (sum(freqz)/len(freqz)))

        #print("rata-rata freq(Hz) : ", meanz)
        #print("selisih rata-rata freq(Hz) : ", -meanz)
        #print("rata-rata freq(Hz) : ", mean2)
        #print("rata-rata freq(Hz) : ", mean2z)
        #print("selisih rata-rata freq(Hz) : ", mean2 - mean2z)
        #print("jumlah LPCF Uji : ", sum(lpc_refeatures))
        #print("Jumlah LPCF Latih : ", sum(lpc_refeaturez))
        #print("selisih LPCF : ", sum(lpc_refeatures) - sum(lpc_refeaturez))
        #print("rata-rata Uji : ", (sum(lpc_refeatures))/16)
        #print("rata-rata Latih : ", (sum(lpc_refeaturez))/16)
        #print("selisih LPCF : ", ((sum(lpc_refeatures))/16) - ((sum(lpc_refeaturez))/16))
        #print("jumlah freq Uji : ", jumlahuji)
        #print("Jumlah freq Latih : ", jumlahlatih)

if label_actual == label_predict:
    status = 'detected'
    if label_predict == 'whatsapp':
        whatsapp_match_value.append(len(freq) - len(freqz))
    elif label_predict == 'linkedin':
        linkedin_match_value.append(len(freq) - len(freqz))
    elif label_predict == 'tokopedia':
        tokopedia_match_value.append(len(freq) - len(freqz))
    elif label_predict == 'gmail':
        gmail_match_value.append(len(freq) - len(freqz))
    elif label_predict == 'powerpoint':
        powerpoint_match_value.append(len(freq) - len(freqz))
    elif label_predict == 'word':
        word_match_value.append(len(freq) - len(freqz))
else:
    if label_predict == 'whatsapp':
        whatsapp_unmatch_value.append(len(freq) - len(freqz))
    elif label_predict == 'linkedin':
        linkedin_unmatch_value.append(len(freq) - len(freqz))
    elif label_predict == 'tokopedia':
        tokopedia_unmatch_value.append(len(freq) - len(freqz))
    elif label_predict == 'gmail':
        gmail_unmatch_value.append(len(freq) - len(freqz))
    elif label_predict == 'powerpoint':
        powerpoint_unmatch_value.append(len(freq) - len(freqz))
    elif label_predict == 'word':
        word_unmatch_value.append(len(freq) - len(freqz))


print("whatsapp match value : ",whatsapp_match_value)
        print("whatsapp unmatch value : ", whatsapp_unmatch_value)
        print("linkedin match value : ", linkedin_match_value)
        print("linkedin unmatch value : ", linkedin_unmatch_value)
        print("tokopedia match value : ", tokopedia_match_value)
        print("tokopedia unmatch value : ", tokopedia_unmatch_value)
        print("gmail match value : ", gmail_match_value)
        print("gmail unmatch value : ", gmail_unmatch_value)
        print("powerpoint match value : ", powerpoint_match_value)
        print("powerpoint unmatch value : ", powerpoint_unmatch_value)
        print("word match value : ", word_match_value)
        print("word unmatch value : ", word_unmatch_value)