using DataFrames

function preProcess()
    d_exp = readtable("exams_metadata_pilot.tsv")
    e_exp = readtable("images_crosswalk_pilot.tsv")

    x_exp = join(d_exp,e_exp,on = :patientId)

    #delete!(x_exp,:daysSincePreviousExam)
    #delete!(x_exp,:implantEver)
    #delete!(x_exp,:implantNow)
    #delete!(x_exp,:breastCancerHistory)
    delete!(x_exp,:yearsSincePreviousBC)
    delete!(x_exp,:previousBCLaterality)
    delete!(x_exp,:reduxHistory)
    delete!(x_exp,:reduxLaterality)
    delete!(x_exp,:hrt)
    delete!(x_exp,:antiestrogen)
    delete!(x_exp,:firstDegreeWithBC)
    delete!(x_exp,:firstDegreeWithBC50)
    delete!(x_exp,:bmi)
    delete!(x_exp,:md5)

    writetable("patient_info.csv",x_exp)
    print(x_exp)
end

function preProcess2()
    dd = readcsv("patient_info.csv")

    for i =2:size(dd)[1]
        dd[i,12] = replace(dd[i,12],".dcm.gz",".dcm")
    end
    writecsv("patient_info_unzip.csv",dd)

end


function preProcess3()
    dd = readtable("patient_info.csv")

    for i =1:size(dd)[1]
        dd[i,12] = replace(dd[i,12],".dcm.gz",".dcm")
    end
    writetable("patient_info_unzip2.csv",dd)

end

preProcess()
    
