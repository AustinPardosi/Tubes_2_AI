# Class App to run the logic of main program
from app.algorithm_manager import AlgorithmManager
import os

class App:
    # Constructor
    def __init__(self):
        self.running = True
        self.algorithm_manager = AlgorithmManager()

    # Main logic of program
    def run(self):
        self.show_header()
        while (self.running):
            self.show_action_options()
            option_num = self.ask_action_options()
            print("========================================================================")

            match option_num:
                case "0":
                    print("Terima kasih telah menggunakan program SimpleLearn")
                    self.running = False
                case "1":
                    filename, load_flag = self.ask_training_options()
                    self.algorithm_manager.train_knn(filename, load_flag)
                    print("Pelatihan selesai dilakukan")
                case "2":
                    filename, load_flag = self.ask_training_options()
                    self.algorithm_manager.train_naive_bayes(filename, load_flag)
                    print("Pelatihan selesai dilakukan")
                case "3":
                    if self.algorithm_manager.is_knn_trained():
                        test_filename, output_filename = self.ask_labelling_options()
                        self.algorithm_manager.classify_with_knn(test_filename, output_filename)
                        print("Labelling selesai dilakukan")
                    else:
                        print("Pastikan anda telah melakukan pelatihan data untuk algoritma KNN")
                case "4":
                    if self.algorithm_manager.is_naive_bayes_trained():
                        test_filename, output_filename = self.ask_labelling_options()
                        self.algorithm_manager.classify_with_naive_bayes(test_filename, output_filename)
                        print("Labelling selesai dilakukan")
                    else:
                        print("Pastikan anda telah melakukan pelatihan data untuk algoritma Naive Bayes")
                case "5":
                    if self.algorithm_manager.is_naive_bayes_trained():
                        output_filename = self.ask_saving_options()
                        self.algorithm_manager.dump_knn(output_filename)
                        print("Model berhasil disimpan.")
                    else:
                        print("Pastikan anda telah melakukan pelatihan data untuk algoritma Naive Bayes")
                case "6":
                    if self.algorithm_manager.is_naive_bayes_trained():
                        output_filename = self.ask_saving_options()
                        self.algorithm_manager.dump_naive_bayes(output_filename)
                        print("Model berhasil disimpan.")
                    else:
                        print("Pastikan anda telah melakukan pelatihan data untuk algoritma Naive Bayes")
                case "7":
                    if self.algorithm_manager.is_naive_bayes_trained():
                        print("Menilai akurasi algoritma KNN memanfaatkan data_validation.csv.")
                        self.algorithm_manager.test_knn_acc()
                    else:
                        print("Pastikan anda telah melakukan pelatihan data untuk algoritma Naive Bayes")
                case "8":
                    if self.algorithm_manager.is_naive_bayes_trained():
                        print("Menilai akurasi algoritma Naive Bayes memanfaatkan data_validation.csv.")
                        self.algorithm_manager.test_naive_bayes_acc()
                    else:
                        print("Pastikan anda telah melakukan pelatihan data untuk algoritma Naive Bayes")
                case _:
                    print("Input tidak valid. Mohon masukkan angka pilihan yang tepat.")
            
            print("========================================================================")

    # Method to show functionality of program
    def show_header(self):
        print("========================================================================")
        print("               Selamat datang di program SimpleLearn                    ")
        print("      Terdapat beberapa fitur yang disediakan oleh program.             ")
        print("========================================================================")

    def show_action_options(self):
        print("Daftar fitur:")
        print("0. Hentikan program")
        print("1. Pelatihan data untuk algoritma KNN")
        print("2. Pelatihan data untuk algoritma Naive Bayes")
        print("3. Labeling data dengan algoritma KNN")
        print("4. Labeling data dengan algoritma Naive Bayes")
        print("5. Simpan model algoritma KNN")
        print("6. Simpan model algoritma Naive Bayes")
        print("7. Lihat akurasi algoritma KNN")
        print("8. Lihat akurasi algoritma Naive Bayes")
        print("========================================================================")
    
    # Method to inquire user about which functionality they want to use
    def ask_action_options(self):
        option = input("Mohon masukkan angka pilihan untuk fitur yang ingin anda gunakan (0-8): ")
        return option
    
    # Method to to ask user the necessary input to do training
    def ask_training_options(self):
        flag_valid = False
        while not flag_valid:
            print("Terdapat beberapa metode pelatihan yang bisa anda gunakan.")
            print("1. Latih dari dataset")
            print("2. Load model dari file pkl")
            method = input("Pilih metode pelatihan yang ingin anda pakai: ")

            if method != "1" and method != "2":
                print("Input tidak valid. Mohon masukkan angka metode pelatihan yang tepat")
            else:
                flag_valid = True

        filename_valid = False
        while not filename_valid:
            filename = input("Mohon masukkan path menuju file (relatif terhadap program ini): ")

            if not os.path.exists(filename):
                print("File tidak ditemukan.")
            elif method == "1" and not filename.endswith(".csv"):
                print("Mohon masukkan file dengan ekstensi csv.")
            elif method == "2" and not filename.endswith(".pkl"):
                print("Mohon masukkan file dengan ekstensi pkl.")
            else:
                filename_valid = True
        
        return filename, method == "2"
    
    # Method to to ask user the necessary input to do labelling
    def ask_labelling_options(self):
        print("Pastikan bahwa file target berada pada folder test.")
        print("File hasil labelling akan diletakkan pada folder result.")

        filename_valid = False
        while not filename_valid:
            test_filename = input("Mohon masukkan nama file target: ")

            if not os.path.exists(f"../test/{test_filename}"):
                print("File tidak ditemukan.")
            elif not test_filename.endswith(".csv"):
                print("Mohon masukkan file dengan ekstensi csv.")
            else:
                filename_valid = True

        filename_valid = False
        while not filename_valid:
            output_filename = input("Mohon masukkan nama file output: ")

            if not output_filename.endswith(".csv"):
                print("Mohon masukkan file dengan ekstensi csv.")
            else:
                filename_valid = True

        return test_filename, output_filename

    # Method to to ask user the necessary input to save model
    def ask_saving_options(self):
        print("File model algoritma nantinya akan diletakkan pada folder model.")

        filename_valid = False
        while not filename_valid:
            output_filename = input("Mohon masukkan nama file target: ")

            if not output_filename.endswith(".pkl"):
                print("Mohon masukkan file dengan ekstensi pkl.")
            else:
                filename_valid = True
        
        return output_filename
        