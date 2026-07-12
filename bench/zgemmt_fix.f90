! Drop-in replacement for the BLAS-extension ZGEMMT (triangular-referenced GEMM:
! C(triangle) := alpha*op(A)*op(B) + beta*C(triangle), other triangle untouched).
! OpenBLAS 0.3.26's zgemmt kernel segfaults when called from ZMUMPS 5.8.2 LDLT
! factorization (even single-threaded); reference netlib BLAS has no GEMMT at all.
! This shim computes column-blocks with ZGEMM into a workspace and copies only the
! referenced triangle. LD_PRELOAD it over libblas so only zgemmt_ is overridden.
subroutine zgemmt(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  implicit none
  character :: uplo, transa, transb
  integer :: n, k, lda, ldb, ldc
  complex*16 :: alpha, beta
  complex*16 :: a(lda,*), b(ldb,*), c(ldc,*)
  integer, parameter :: nbmax = 192
  complex*16, allocatable :: w(:,:)
  complex*16, parameter :: zone = (1.0d0, 0.0d0), zzero = (0.0d0, 0.0d0)
  integer :: j1, j2, jb, jj, i, m1, r0, jg
  logical :: upper, notra
  logical, external :: lsame
  external :: zgemm

  if (n <= 0) return
  upper = lsame(uplo, 'U')
  notra = lsame(transa, 'N')

  if (alpha == zzero .or. k <= 0) then  ! only scale the referenced triangle
    do jj = 1, n
      do i = merge(1, jj, upper), merge(jj, n, upper)
        if (beta == zzero) then
          c(i, jj) = zzero
        else
          c(i, jj) = beta * c(i, jj)
        end if
      end do
    end do
    return
  end if

  allocate(w(n, min(n, nbmax)))
  do j1 = 1, n, nbmax
    jb = min(nbmax, n - j1 + 1)
    j2 = j1 + jb - 1
    if (upper) then
      m1 = j2       ! need rows 1..j2 of the product for these columns
      r0 = 1
    else
      m1 = n - j1 + 1  ! need rows j1..n
      r0 = j1
    end if
    ! W(1:m1, 1:jb) = op(A)(r0:r0+m1-1, :) * op(B)(:, j1:j2)
    if (notra) then
      if (lsame(transb, 'N')) then
        call zgemm(transa, transb, m1, jb, k, zone, a(r0, 1), lda, b(1, j1), ldb, zzero, w, n)
      else
        call zgemm(transa, transb, m1, jb, k, zone, a(r0, 1), lda, b(j1, 1), ldb, zzero, w, n)
      end if
    else
      if (lsame(transb, 'N')) then
        call zgemm(transa, transb, m1, jb, k, zone, a(1, r0), lda, b(1, j1), ldb, zzero, w, n)
      else
        call zgemm(transa, transb, m1, jb, k, zone, a(1, r0), lda, b(j1, 1), ldb, zzero, w, n)
      end if
    end if
    do jj = 1, jb
      jg = j1 + jj - 1  ! global column
      do i = merge(1, jg, upper), merge(jg, n, upper)
        if (beta == zzero) then
          c(i, jg) = alpha * w(merge(i, i - j1 + 1, upper), jj)
        else
          c(i, jg) = alpha * w(merge(i, i - j1 + 1, upper), jj) + beta * c(i, jg)
        end if
      end do
    end do
  end do
  deallocate(w)
end subroutine zgemmt
